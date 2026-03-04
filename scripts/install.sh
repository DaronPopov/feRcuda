#!/usr/bin/env bash
set -euo pipefail

PREFIX="${HOME}/.local"
BUILD_TYPE="Release"
ARCH_LIST=""
WITH_TESTS=0
JOBS="${JOBS:-$(nproc)}"
REPO_URL="${FERCUDA_REPO_URL:-https://github.com/DaronPopov/feRcuda.git}"
KEEP_WORKDIR=0

print_help() {
  cat <<'USAGE'
feR-os installer (feRcuda)

Usage:
  install.sh [options]

Options:
  --prefix <path>       Install prefix (default: ~/.local)
  --build-type <type>   CMake build type (default: Release)
  --arch-list <list>    CUDA arch list for CMAKE_CUDA_ARCHITECTURES (e.g. "75;86;89")
  --with-tests          Build and run tests after compile
  --jobs <n>            Parallel build jobs (default: nproc)
  --repo-url <url>      Git URL used when bootstrapping (default: https://github.com/DaronPopov/feRcuda.git)
  --keep-workdir        Keep temporary cloned directory
  -h, --help            Show this help
USAGE
}

while [[ $# -gt 0 ]]; do
  case "$1" in
    --prefix)
      PREFIX="$2"
      shift 2
      ;;
    --build-type)
      BUILD_TYPE="$2"
      shift 2
      ;;
    --arch-list)
      ARCH_LIST="$2"
      shift 2
      ;;
    --with-tests)
      WITH_TESTS=1
      shift
      ;;
    --jobs)
      JOBS="$2"
      shift 2
      ;;
    --repo-url)
      REPO_URL="$2"
      shift 2
      ;;
    --keep-workdir)
      KEEP_WORKDIR=1
      shift
      ;;
    -h|--help)
      print_help
      exit 0
      ;;
    *)
      echo "Unknown arg: $1" >&2
      print_help
      exit 1
      ;;
  esac
done

need_cmd() {
  command -v "$1" >/dev/null 2>&1 || {
    echo "Missing required command: $1" >&2
    exit 1
  }
}

need_cmd cmake
need_cmd git

if ! command -v nvcc >/dev/null 2>&1; then
  echo "nvcc not found in PATH. Install CUDA toolkit before running installer." >&2
  exit 1
fi

resolve_repo_root() {
  if [[ -f "./CMakeLists.txt" && -d "./src" && -d "./include" ]]; then
    pwd
    return 0
  fi

  local workdir
  workdir="$(mktemp -d /tmp/feros-install-XXXXXX)"
  echo "[feR-os] cloning ${REPO_URL} -> ${workdir}"
  git clone --depth 1 "${REPO_URL}" "${workdir}/feRcuda"
  echo "${workdir}/feRcuda"
}

ROOT_DIR="$(resolve_repo_root)"
BUILD_DIR="${ROOT_DIR}/build"

if [[ "${KEEP_WORKDIR}" -eq 0 && "${ROOT_DIR}" == /tmp/feros-install-*/* ]]; then
  TMP_PARENT="$(dirname "${ROOT_DIR}")"
  trap 'rm -rf "${TMP_PARENT}"' EXIT
fi

mkdir -p "${BUILD_DIR}"

CMAKE_ARGS=(
  -S "${ROOT_DIR}"
  -B "${BUILD_DIR}"
  -DCMAKE_BUILD_TYPE="${BUILD_TYPE}"
)

if [[ -n "${ARCH_LIST}" ]]; then
  CMAKE_ARGS+=( -DCMAKE_CUDA_ARCHITECTURES="${ARCH_LIST}" )
fi

echo "[feR-os] configure"
cmake "${CMAKE_ARGS[@]}"

echo "[feR-os] build"
cmake --build "${BUILD_DIR}" -j"${JOBS}"

echo "[feR-os] install -> ${PREFIX}"
cmake --install "${BUILD_DIR}" --prefix "${PREFIX}"

if [[ "${WITH_TESTS}" -eq 1 ]]; then
  echo "[feR-os] test"
  (cd "${BUILD_DIR}" && ctest --output-on-failure)
fi

echo "[feR-os] done"
echo "Export runtime path if needed:"
echo "  export LD_LIBRARY_PATH=\"${PREFIX}/lib:\${LD_LIBRARY_PATH:-}\""
