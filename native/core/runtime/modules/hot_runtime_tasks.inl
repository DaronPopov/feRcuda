// ============================================================================
// Task Submission API
// ============================================================================

int ptx_os_submit_task(GPUHotRuntime* runtime, uint32_t opcode, uint32_t priority, void* args[PTX_MAX_TASK_ARGS]) {
    if (!runtime || !runtime->system_state) return -1;

    PTXSystemState* state = runtime->system_state;
    PTXTaskQueue* queue = &state->queue;

    // Spinlock acquire
    while (__sync_lock_test_and_set(&queue->lock, 1)) {
        // spin
    }

    uint32_t next_head = (queue->head + 1) % PTX_MAX_QUEUE_SIZE;
    if (next_head == queue->tail) {
        // Queue full
        __sync_lock_release(&queue->lock);
        return -1;
    }

    PTXOSTask* task = &queue->tasks[queue->head];
    task->task_id = queue->head;
    task->opcode = opcode;
    task->priority = priority;
    task->active = true;
    task->completed = false;
    if (args) {
        memcpy(task->args, args, sizeof(void*) * PTX_MAX_TASK_ARGS);
    } else {
        memset(task->args, 0, sizeof(void*) * PTX_MAX_TASK_ARGS);
    }

    __sync_synchronize();
    queue->head = next_head;

    __sync_lock_release(&queue->lock);

    return (int)task->task_id;
}
