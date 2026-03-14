//! PTX-OS Daemon - Persistent GPU Runtime Service
//!
//! Keeps the GPU runtime hot and programmable via Unix socket IPC.
//!
//! Usage:
//!   ptx_daemon start          # Start the daemon
//!   ptx_daemon status         # Check daemon status
//!   ptx_daemon stats          # Get pool statistics
//!   ptx_daemon alloc <size>   # Allocate memory (returns handle)
//!   ptx_daemon free <handle>  # Free memory
//!   ptx_daemon stop           # Graceful shutdown

use ptx_os::RegimeRuntimeCore;
use std::collections::HashMap;
use std::fs;
use std::io::{BufRead, BufReader, Read, Write};
use std::os::unix::net::{UnixListener, UnixStream};
use std::path::Path;
use std::sync::atomic::{AtomicBool, AtomicU64, Ordering};
use std::sync::Arc;
use std::thread;
use std::time::{Duration, Instant};

const SOCKET_PATH: &str = "/tmp/ptx_daemon.sock";
const PID_FILE: &str = "/tmp/ptx_daemon.pid";

/// Daemon state
struct DaemonState {
    runtime: RegimeRuntimeCore,
    allocations: parking_lot::Mutex<HashMap<u64, Allocation>>,
    next_handle: AtomicU64,
    start_time: Instant,
    total_ops: AtomicU64,
    running: AtomicBool,
}

struct Allocation {
    ptr: *mut std::ffi::c_void,
    size: usize,
    created_at: Instant,
}

// Safety: We manage the pointers carefully within the daemon
unsafe impl Send for Allocation {}
unsafe impl Sync for Allocation {}

impl DaemonState {
    fn new(device_id: i32) -> Result<Self, Box<dyn std::error::Error>> {
        let runtime = RegimeRuntimeCore::new(device_id)?;

        Ok(Self {
            runtime,
            allocations: parking_lot::Mutex::new(HashMap::new()),
            next_handle: AtomicU64::new(1),
            start_time: Instant::now(),
            total_ops: AtomicU64::new(0),
            running: AtomicBool::new(true),
        })
    }

    fn alloc(&self, size: usize) -> Result<u64, String> {
        let ptr = self.runtime.alloc_raw(size)
            .map_err(|e| format!("Allocation failed: {:?}", e))?;

        let handle = self.next_handle.fetch_add(1, Ordering::Relaxed);
        let alloc = Allocation {
            ptr,
            size,
            created_at: Instant::now(),
        };

        self.allocations.lock().insert(handle, alloc);
        self.total_ops.fetch_add(1, Ordering::Relaxed);

        Ok(handle)
    }

    fn free(&self, handle: u64) -> Result<(), String> {
        let alloc = self.allocations.lock().remove(&handle)
            .ok_or_else(|| format!("Invalid handle: {}", handle))?;

        unsafe {
            self.runtime.free_raw(alloc.ptr);
        }
        self.total_ops.fetch_add(1, Ordering::Relaxed);

        Ok(())
    }

    fn stats(&self) -> DaemonStats {
        let pool = self.runtime.pool_stats();
        let allocs = self.allocations.lock();

        DaemonStats {
            uptime_secs: self.start_time.elapsed().as_secs(),
            total_ops: self.total_ops.load(Ordering::Relaxed),
            active_allocations: allocs.len(),
            pool_total_mb: pool.total_size as f64 / (1024.0 * 1024.0),
            pool_allocated_mb: pool.allocated as f64 / (1024.0 * 1024.0),
            pool_free_mb: pool.free as f64 / (1024.0 * 1024.0),
            utilization_pct: pool.utilization_percent,
            fragmentation: pool.fragmentation_ratio,
            is_healthy: pool.is_healthy,
        }
    }

    fn list_allocations(&self) -> Vec<(u64, usize, u64)> {
        let allocs = self.allocations.lock();
        allocs.iter()
            .map(|(&h, a)| (h, a.size, a.created_at.elapsed().as_secs()))
            .collect()
    }

    fn defragment(&self) {
        self.runtime.defragment();
        self.total_ops.fetch_add(1, Ordering::Relaxed);
    }

    fn keepalive(&self) {
        self.runtime.keepalive();
    }
}

#[derive(Debug)]
struct DaemonStats {
    uptime_secs: u64,
    total_ops: u64,
    active_allocations: usize,
    pool_total_mb: f64,
    pool_allocated_mb: f64,
    pool_free_mb: f64,
    utilization_pct: f32,
    fragmentation: f32,
    is_healthy: bool,
}

impl std::fmt::Display for DaemonStats {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f,
            "PTX-OS Daemon Statistics\n\
             ========================\n\
             Uptime:           {} seconds\n\
             Total Operations: {}\n\
             Active Allocs:    {}\n\
             \n\
             Memory Pool:\n\
             Total:            {:.2} MB\n\
             Allocated:        {:.2} MB\n\
             Free:             {:.2} MB\n\
             Utilization:      {:.1}%\n\
             Fragmentation:    {:.1}%\n\
             Health:           {}",
            self.uptime_secs,
            self.total_ops,
            self.active_allocations,
            self.pool_total_mb,
            self.pool_allocated_mb,
            self.pool_free_mb,
            self.utilization_pct,
            self.fragmentation * 100.0,
            if self.is_healthy { "OK" } else { "DEGRADED" }
        )
    }
}

fn handle_command(state: &DaemonState, cmd: &str) -> String {
    let parts: Vec<&str> = cmd.trim().split_whitespace().collect();
    if parts.is_empty() {
        return "ERROR: Empty command".to_string();
    }

    match parts[0].to_lowercase().as_str() {
        "ping" => "PONG".to_string(),

        "status" => format!("OK: Daemon running, uptime {} secs",
            state.start_time.elapsed().as_secs()),

        "stats" => format!("{}", state.stats()),

        "alloc" => {
            if parts.len() < 2 {
                return "ERROR: Usage: alloc <size_bytes>".to_string();
            }
            match parts[1].parse::<usize>() {
                Ok(size) => match state.alloc(size) {
                    Ok(handle) => format!("OK: handle={}", handle),
                    Err(e) => format!("ERROR: {}", e),
                },
                Err(_) => "ERROR: Invalid size".to_string(),
            }
        }

        "free" => {
            if parts.len() < 2 {
                return "ERROR: Usage: free <handle>".to_string();
            }
            match parts[1].parse::<u64>() {
                Ok(handle) => match state.free(handle) {
                    Ok(()) => "OK: Freed".to_string(),
                    Err(e) => format!("ERROR: {}", e),
                },
                Err(_) => "ERROR: Invalid handle".to_string(),
            }
        }

        "list" => {
            let allocs = state.list_allocations();
            if allocs.is_empty() {
                "OK: No active allocations".to_string()
            } else {
                let lines: Vec<String> = allocs.iter()
                    .map(|(h, s, age)| format!("  handle={} size={} age={}s", h, s, age))
                    .collect();
                format!("OK: {} allocations\n{}", allocs.len(), lines.join("\n"))
            }
        }

        "defrag" => {
            state.defragment();
            "OK: Defragmentation triggered".to_string()
        }

        "gc" => {
            // Garbage collect old allocations (> 1 hour)
            let mut freed = 0;
            let handles: Vec<u64> = state.allocations.lock()
                .iter()
                .filter(|(_, a)| a.created_at.elapsed() > Duration::from_secs(3600))
                .map(|(&h, _)| h)
                .collect();

            for handle in handles {
                if state.free(handle).is_ok() {
                    freed += 1;
                }
            }
            format!("OK: Garbage collected {} allocations", freed)
        }

        "shutdown" | "stop" => {
            state.running.store(false, Ordering::SeqCst);
            "OK: Shutdown initiated".to_string()
        }

        "stream" => {
            // Hot streaming benchmark - continuous alloc/process/free cycles
            let iterations = parts.get(1).and_then(|s| s.parse().ok()).unwrap_or(1000);
            let size = parts.get(2).and_then(|s| s.parse().ok()).unwrap_or(65536); // 64KB default

            let start = Instant::now();
            let mut total_bytes = 0usize;
            let mut min_lat = u128::MAX;
            let mut max_lat = 0u128;
            let mut total_lat = 0u128;

            for _ in 0..iterations {
                let iter_start = Instant::now();

                // Allocate
                let handle = match state.alloc(size) {
                    Ok(h) => h,
                    Err(e) => return format!("ERROR: Stream failed: {}", e),
                };

                // "Process" - just touch the memory (in real use, kernel would run here)
                // The allocation itself exercises the hot path

                // Free
                if let Err(e) = state.free(handle) {
                    return format!("ERROR: Stream free failed: {}", e);
                }

                let lat = iter_start.elapsed().as_nanos();
                min_lat = min_lat.min(lat);
                max_lat = max_lat.max(lat);
                total_lat += lat;
                total_bytes += size;
            }

            let elapsed = start.elapsed();
            let throughput_mb = (total_bytes as f64 / (1024.0 * 1024.0)) / elapsed.as_secs_f64();
            let avg_lat = total_lat / iterations as u128;

            format!(
                "Streaming Benchmark Results\n\
                 ===========================\n\
                 Iterations:    {}\n\
                 Chunk Size:    {} bytes\n\
                 Total Data:    {:.2} MB\n\
                 Duration:      {:.2} ms\n\
                 Throughput:    {:.2} MB/s\n\
                 \n\
                 Latency (per iteration):\n\
                 Min:           {} ns\n\
                 Avg:           {} ns\n\
                 Max:           {} ns\n\
                 \n\
                 Status:        GPU IS HOT",
                iterations,
                size,
                total_bytes as f64 / (1024.0 * 1024.0),
                elapsed.as_secs_f64() * 1000.0,
                throughput_mb,
                min_lat,
                avg_lat,
                max_lat
            )
        }

        "burst" => {
            // Burst allocation test - allocate many, then free all
            let count = parts.get(1).and_then(|s| s.parse().ok()).unwrap_or(100);
            let size = parts.get(2).and_then(|s| s.parse().ok()).unwrap_or(1048576); // 1MB default

            let mut handles = Vec::with_capacity(count);

            // Allocate burst
            let alloc_start = Instant::now();
            for _ in 0..count {
                match state.alloc(size) {
                    Ok(h) => handles.push(h),
                    Err(e) => {
                        // Free what we allocated
                        let allocated = handles.len();
                        for h in handles {
                            let _ = state.free(h);
                        }
                        return format!("ERROR: Burst alloc failed after {}: {}", allocated, e);
                    }
                }
            }
            let alloc_time = alloc_start.elapsed();
            let allocated_count = handles.len();

            // Free burst
            let free_start = Instant::now();
            for h in handles {
                let _ = state.free(h);
            }
            let free_time = free_start.elapsed();

            let total_mb = (allocated_count * size) as f64 / (1024.0 * 1024.0);

            format!(
                "Burst Allocation Results\n\
                 ========================\n\
                 Allocations:   {}\n\
                 Size Each:     {} bytes\n\
                 Total:         {:.2} MB\n\
                 \n\
                 Alloc Time:    {:.2} ms ({:.0} ns/alloc)\n\
                 Free Time:     {:.2} ms ({:.0} ns/free)\n\
                 \n\
                 Alloc Rate:    {:.2} MB/s\n\
                 Free Rate:     {:.2} MB/s",
                allocated_count,
                size,
                total_mb,
                alloc_time.as_secs_f64() * 1000.0,
                alloc_time.as_nanos() as f64 / allocated_count as f64,
                free_time.as_secs_f64() * 1000.0,
                free_time.as_nanos() as f64 / allocated_count as f64,
                total_mb / alloc_time.as_secs_f64(),
                total_mb / free_time.as_secs_f64()
            )
        }

        "help" => {
            "PTX-OS Daemon Commands:\n\
             ping              - Check if daemon is alive\n\
             status            - Get daemon status\n\
             stats             - Get detailed statistics\n\
             alloc <size>      - Allocate GPU memory (returns handle)\n\
             free <handle>     - Free allocation by handle\n\
             list              - List active allocations\n\
             defrag            - Trigger defragmentation\n\
             gc                - Garbage collect old allocations\n\
             stream [n] [size] - Streaming benchmark (n iterations, size bytes)\n\
             burst [n] [size]  - Burst alloc/free test (n allocations)\n\
             shutdown          - Graceful shutdown\n\
             help              - Show this help".to_string()
        }

        _ => format!("ERROR: Unknown command '{}'. Type 'help' for commands.", parts[0]),
    }
}

fn handle_client(state: Arc<DaemonState>, mut stream: UnixStream) {
    let mut reader = BufReader::new(stream.try_clone().expect("Failed to clone stream"));
    let mut cmd = String::new();

    // Read single command
    if reader.read_line(&mut cmd).is_err() {
        return;
    }

    let response = handle_command(&state, &cmd);

    // Send response and close
    let _ = writeln!(stream, "{}", response);
    let _ = stream.flush();
    let _ = stream.shutdown(std::net::Shutdown::Both);
}

fn run_daemon(device_id: i32) -> Result<(), Box<dyn std::error::Error>> {
    // Remove stale socket
    if Path::new(SOCKET_PATH).exists() {
        fs::remove_file(SOCKET_PATH)?;
    }

    // Initialize GPU runtime
    println!("[ptx_daemon] Initializing GPU runtime on device {}...", device_id);
    let state = Arc::new(DaemonState::new(device_id)?);

    // Write PID file
    fs::write(PID_FILE, format!("{}", std::process::id()))?;

    // Create socket
    let listener = UnixListener::bind(SOCKET_PATH)?;
    listener.set_nonblocking(true)?;

    println!("[ptx_daemon] Listening on {}", SOCKET_PATH);
    println!("[ptx_daemon] GPU is HOT and ready!");
    println!("[ptx_daemon] Press Ctrl+C or send 'shutdown' to stop.");

    // Keepalive thread
    let keepalive_state = Arc::clone(&state);
    let keepalive_handle = thread::spawn(move || {
        while keepalive_state.running.load(Ordering::SeqCst) {
            keepalive_state.keepalive();
            thread::sleep(Duration::from_secs(30));
        }
    });

    // Main accept loop
    while state.running.load(Ordering::SeqCst) {
        match listener.accept() {
            Ok((stream, _)) => {
                let client_state = Arc::clone(&state);
                thread::spawn(move || {
                    handle_client(client_state, stream);
                });
            }
            Err(ref e) if e.kind() == std::io::ErrorKind::WouldBlock => {
                thread::sleep(Duration::from_millis(100));
            }
            Err(e) => {
                eprintln!("[daemon] Accept error: {}", e);
            }
        }
    }

    println!("[ptx_daemon] Shutting down...");

    // Wait for keepalive thread
    let _ = keepalive_handle.join();

    // Cleanup
    let _ = fs::remove_file(SOCKET_PATH);
    let _ = fs::remove_file(PID_FILE);

    // Free any remaining allocations
    let handles: Vec<u64> = state.allocations.lock().keys().copied().collect();
    for handle in handles {
        let _ = state.free(handle);
    }

    println!("[ptx_daemon] Goodbye!");
    Ok(())
}

fn send_command(cmd: &str) -> Result<String, Box<dyn std::error::Error>> {
    let mut stream = UnixStream::connect(SOCKET_PATH)?;
    stream.set_read_timeout(Some(Duration::from_secs(5)))?;

    writeln!(stream, "{}", cmd)?;
    stream.flush()?;

    // Shutdown write side to signal we're done sending
    stream.shutdown(std::net::Shutdown::Write)?;

    let mut response = String::new();
    BufReader::new(stream).read_to_string(&mut response)?;

    Ok(response.trim().to_string())
}

fn is_daemon_running() -> bool {
    Path::new(SOCKET_PATH).exists() && send_command("ping").is_ok()
}

fn print_usage() {
    println!("PTX-OS Daemon - Persistent GPU Runtime Service");
    println!();
    println!("Usage: ptx_daemon <command> [args]");
    println!();
    println!("Commands:");
    println!("  start [device]  Start the daemon (default device: 0)");
    println!("  stop            Stop the daemon");
    println!("  status          Check daemon status");
    println!("  stats           Get pool statistics");
    println!("  alloc <size>    Allocate GPU memory");
    println!("  free <handle>   Free allocation");
    println!("  list            List active allocations");
    println!("  defrag          Trigger defragmentation");
    println!("  gc              Garbage collect old allocations");
    println!();
    println!("Streaming/Benchmark:");
    println!("  stream [n] [size]   Hot streaming test (default: 1000 x 64KB)");
    println!("  burst [n] [size]    Burst alloc/free test (default: 100 x 1MB)");
    println!();
    println!("Examples:");
    println!("  ptx_daemon start           # Start daemon on GPU 0");
    println!("  ptx_daemon stream 10000    # 10K streaming iterations");
    println!("  ptx_daemon burst 500 4096  # 500 x 4KB burst test");
}

fn main() {
    let args: Vec<String> = std::env::args().collect();

    if args.len() < 2 {
        print_usage();
        return;
    }

    let result = match args[1].as_str() {
        "start" => {
            if is_daemon_running() {
                println!("Daemon is already running!");
                return;
            }
            let device_id = args.get(2)
                .and_then(|s| s.parse().ok())
                .unwrap_or(0);
            run_daemon(device_id)
        }

        "stop" | "shutdown" => {
            send_command("shutdown").map(|r| println!("{}", r))
        }

        "status" => {
            if is_daemon_running() {
                send_command("status").map(|r| println!("{}", r))
            } else {
                println!("Daemon is not running.");
                Ok(())
            }
        }

        "stats" => send_command("stats").map(|r| println!("{}", r)),
        "list" => send_command("list").map(|r| println!("{}", r)),
        "defrag" => send_command("defrag").map(|r| println!("{}", r)),
        "gc" => send_command("gc").map(|r| println!("{}", r)),

        "stream" => {
            let n = args.get(2).unwrap_or(&"1000".to_string()).clone();
            let size = args.get(3).unwrap_or(&"65536".to_string()).clone();
            send_command(&format!("stream {} {}", n, size)).map(|r| println!("{}", r))
        }

        "burst" => {
            let n = args.get(2).unwrap_or(&"100".to_string()).clone();
            let size = args.get(3).unwrap_or(&"1048576".to_string()).clone();
            send_command(&format!("burst {} {}", n, size)).map(|r| println!("{}", r))
        }

        "alloc" => {
            if args.len() < 3 {
                println!("Usage: ptx_daemon alloc <size_bytes>");
                return;
            }
            send_command(&format!("alloc {}", args[2])).map(|r| println!("{}", r))
        }

        "free" => {
            if args.len() < 3 {
                println!("Usage: ptx_daemon free <handle>");
                return;
            }
            send_command(&format!("free {}", args[2])).map(|r| println!("{}", r))
        }

        "ping" => send_command("ping").map(|r| println!("{}", r)),

        "help" | "-h" | "--help" => {
            print_usage();
            Ok(())
        }

        cmd => {
            // Try to send as raw command
            send_command(cmd).map(|r| println!("{}", r))
        }
    };

    if let Err(e) = result {
        if e.to_string().contains("No such file") || e.to_string().contains("Connection refused") {
            eprintln!("Error: Daemon is not running. Start it with: ptx_daemon start");
        } else {
            eprintln!("Error: {}", e);
        }
        std::process::exit(1);
    }
}
