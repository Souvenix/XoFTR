@echo off
SETLOCAL

:: 获取脚本所在目录
set SCRIPT_DIR=%~dp0
set PROJECT_DIR=%SCRIPT_DIR%..\..

:: 设置环境变量 PYTHONPATH
set PYTHONPATH=%PROJECT_DIR%;%PYTHONPATH%

:: 切换到项目根目录
cd /d %PROJECT_DIR%

:: 配置路径
set DATA_CFG_PATH=configs/data/pretrain.py
set MAIN_CFG_PATH=configs/xoftr/pretrain/pretrain.py

:: 训练参数配置
set NUM_WORKERS=1
set BATCH_SIZE=1
set PIN_MEMORY=True
set EXP_NAME=pretrain-%%TRAIN_IMG_SIZE%%-bs=1_1_%BATCH_SIZE%

:: 启动训练
python -u ./pretrain.py ^
    %DATA_CFG_PATH% ^
    %MAIN_CFG_PATH% ^
    --exp_name=%EXP_NAME% ^
    --gpus=1 ^
    --num_nodes=1 ^
    --accelerator=gpu ^
    --batch_size=%BATCH_SIZE% ^
    --num_workers=%NUM_WORKERS% ^
    --pin_memory=%PIN_MEMORY% ^
    --check_val_every_n_epoch=1 ^
    --log_every_n_steps=100 ^
    --limit_val_batches=1.0 ^
    --num_sanity_val_steps=10 ^
    --benchmark=True ^
    --max_epochs=15 ^
    --ckpt_path weights/weights_xoftr_640.ckpt

ENDLOCAL
