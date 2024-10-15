@echo off
set BASE_URL=https://dl.fbaipublicfiles.com/segment_anything_2/072824/

set sam2_hiera_t_url=%BASE_URL%sam2_hiera_tiny.pt
set sam2_hiera_s_url=%BASE_URL%sam2_hiera_small.pt
set sam2_hiera_b_plus_url=%BASE_URL%sam2_hiera_base_plus.pt
set sam2_hiera_l_url=%BASE_URL%sam2_hiera_large.pt

echo Downloading sam2_hiera_tiny.pt checkpoint...
wget %sam2_hiera_t_url%
if %errorlevel% neq 0 (
    echo Failed to download checkpoint from %sam2_hiera_t_url%
    exit /b 1
)

echo Downloading sam2_hiera_small.pt checkpoint...
wget %sam2_hiera_s_url%
if %errorlevel% neq 0 (
    echo Failed to download checkpoint from %sam2_hiera_s_url%
    exit /b 1
)

echo Downloading sam2_hiera_base_plus.pt checkpoint...
wget %sam2_hiera_b_plus_url%
if %errorlevel% neq 0 (
    echo Failed to download checkpoint from %sam2_hiera_b_plus_url%
    exit /b 1
)

echo Downloading sam2_hiera_large.pt checkpoint...
wget %sam2_hiera_l_url%
if %errorlevel% neq 0 (
    echo Failed to download checkpoint from %sam2_hiera_l_url%
    exit /b 1
)

echo All checkpoints are downloaded successfully.
