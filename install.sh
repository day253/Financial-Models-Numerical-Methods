# /bin/
BASE_DIR=$(
    cd $(dirname "$0")
    pwd
)

# 安装venv
make

# 安装cpython扩展
cd $BASE_DIR/functions/C
make

cd $BASE_DIR/functions/cython

NUMPY_INCLUDE=$(python -c '
import numpy as np
print(np.get_include())
')

export CFLAGS="-I ${NUMPY_INCLUDE} $CFLAGS"
python setup.py build_ext --inplace
