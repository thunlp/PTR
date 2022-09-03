if [ $# -lt 1 ]; then
    echo "At least one dataset should be specified. Or you can download all datasets by running this script with the argument 'all'."
    exit
fi

declare -A links=( 
    ["retacred"]="https://cloud.tsinghua.edu.cn/f/6b8489ca564544a38452/?dl=1" 
    ["tacred"]="https://cloud.tsinghua.edu.cn/f/51e7565126854e49a239/?dl=1" 
    ["tacrev"]="https://cloud.tsinghua.edu.cn/f/f2423f0a104248d0a2de/?dl=1" 
    ["semeval"]="https://cloud.tsinghua.edu.cn/f/fff36d74ec93427ca5aa/?dl=1"
)

download_all_flag=false
for dataset in "$@" 
do
    if [ $dataset = "all" ]; then
        download_all_flag=true
        break
    fi
done

if [ $download_all_flag = true ]; then
    DATASETS=(retacred tacred tacrev semeval)
else
    DATASETS=$@
fi

echo "Downloading datasets: ${DATASETS[@]}"

for dataset in "${DATASETS[@]}"; do
    wget --no-check-certificate -O data/$dataset.tgz ${links[$dataset]}
    tar -xzvf data/${dataset}.tgz -C data/
    rm data/${dataset}.tgz
done