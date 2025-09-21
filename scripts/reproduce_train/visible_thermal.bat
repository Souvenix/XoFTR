@echo off
SETLOCAL

:: 获取脚本所在目录
set SCRIPT_DIR=%~dp0
set PROJECT_DIR=%SCRIPT_DIR%..\..

:: 设置环境变量 PYTHONPATH - Windows使用set而不是export
set PYTHONPATH=%PROJECT_DIR%;%PYTHONPATH%

:: 添加Windows兼容的分布式后端设置
set PL_TORCH_DISTRIBUTED_BACKEND=gloo

:: 切换到项目根目录
cd /d %PROJECT_DIR%

:: 训练图像大小配置
set TRAIN_IMG_SIZE=640
:: set TRAIN_IMG_SIZE=840
set DATA_CFG_PATH=configs\data\megadepth_vistir_trainval_%TRAIN_IMG_SIZE%.py
set MAIN_CFG_PATH=configs\xoftr\outdoor\visible_thermal.py

:: 训练参数配置 - 使用1个GPU不需要分布式训练
set N_NODES=1
set N_GPUS_PER_NODE=1
set TORCH_NUM_WORKERS=1
set BATCH_SIZE=2
set PIN_MEMORY=true
set CKPT_PATH=weights\weights_xoftr_640.ckpt

:: 计算总批量大小并设置实验名称
set /a TOTAL_BATCH_SIZE=%N_GPUS_PER_NODE% * %N_NODES% * %BATCH_SIZE%
set EXP_NAME=visible_thermal-%TRAIN_IMG_SIZE%-bs=%TOTAL_BATCH_SIZE%

:: 启动训练 - 使用单GPU策略，不指定分布式策略
python -u ./train.py ^
    %DATA_CFG_PATH% ^
    %MAIN_CFG_PATH% ^
    --exp_name=%EXP_NAME% ^
    --gpus=%N_GPUS_PER_NODE% --num_nodes=%N_NODES% ^
    --batch_size=%BATCH_SIZE% --num_workers=%TORCH_NUM_WORKERS% --pin_memory=%PIN_MEMORY% ^
    --check_val_every_n_epoch=1 ^
    --log_every_n_steps=100 ^
    --limit_val_batches=1. ^
    --num_sanity_val_steps=10 ^
    --benchmark=True ^
    --max_epochs=30 ^
    --ckpt_path=%CKPT_PATH%

ENDLOCAL