#!/bin/bash

# 使用示例：./npu_monitor_max.sh [循环次数] [间隔秒数]
# 示例：监控20周期，每10秒 → ./npu_monitor_max.sh 20 10

# 配置参数
CUR_CARDS=${1:-1}           # NPU卡
# TOTAL_CARDS=${1:-8}           # NPU卡总数
INTERVAL=${2:-5}       # 默认5秒间隔


# 初始化最大值记录
MAX_AICORE=0
MAX_HBM=0
MAX_BANDWIDTH=0

# 捕获退出信号
trap "echo '收到停止信号，结束执行'; exit" SIGTERM SIGINT
echo "开始持续执行，PID=$$"
echo "要停止请运行: kill $$ 或 kill -INT $$"

# 参数校验函数
validate_number() {
    [[ "$1" =~ ^[0-9]+$ ]] || {
        echo "错误：参数必须为整数";
        echo "用法：$0 [循环次数] [间隔秒数]";
        exit 1
    }
}

# 执行参数校验
validate_number $CUR_CARDS
validate_number $INTERVAL

# 打印监控头
echo "NPU集群监控 (含最大值追踪)"
# echo "设备卡数:$TOTAL_CARDS 间隔:${INTERVAL}s"
echo "设备卡数:$CUR_CARDS 间隔:${INTERVAL}s"
printf "%-12s | %-9s | %-9s | %-9s | %-9s \n" "时间戳" "算力(当前/最大)" "显存(当前/最大)" "带宽(当前/最大)" "内存(当前/最大)"


# 主监控循环
while true;
do
    # 初始化累计值
    aicore_total=0
    hbm_usage_total=0
    bandwidth_total=0

    # 全卡数据采集
    for ((card=CUR_CARDS; card<CUR_CARDS+1; card++))
    do
        # usage_data=$(nvidia-smi -i $card -t usages 2>/dev/null)
        usage_data=$(nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits -i $card 2>/dev/null)
        mem_used=$(( usage_data * 100 / 81920 ))
        TIMESTAMP=$(date '+%H:%M:%S')
        echo "[$TIMESTAMP]"
        echo "card $card: $usage_data  $mem_used%"
        
        # # 数据解析（带错误抑制）
        # aicore=$(    { grep "Aicore Usage" <<< "$usage_data" || echo "0"; } | awk -F': ' '{print $2}' | cut -d'%' -f1)
        # hbm_usage=$( { grep "HBM Usage" <<< "$usage_data"    || echo "0"; } | awk -F': ' '{print $2}' | cut -d'%' -f1)
        # hbm_bw=$(    { grep "HBM Bandwidth" <<< "$usage_data"|| echo "0"; } | awk -F': ' '{print $2}' | cut -d'%' -f1)

        # # 数值累加
        # aicore_total=$(awk -v total="$aicore_total" -v add="${aicore:-0}" 'BEGIN {printf "%.1f", total + add}')
        # hbm_usage_total=$(awk -v total="$hbm_usage_total" -v add="${hbm_usage:-0}" 'BEGIN {printf "%.1f", total + add}')
        # bandwidth_total=$(awk -v total="$bandwidth_total" -v add="${hbm_bw:-0}" 'BEGIN {printf "%.1f", total + add}')

    done
    
    # # 计算当前周期平均值
    # aicore_avg=$(awk -v total="$aicore_total" -v cards="$TOTAL_CARDS" 'BEGIN {printf "%.1f", total / cards}')
    # hbm_avg=$(awk -v total="$hbm_usage_total" -v cards="$TOTAL_CARDS" 'BEGIN {printf "%.1f", total / cards}')
    # bandwidth_avg=$(awk -v total="$bandwidth_total" -v cards="$TOTAL_CARDS" 'BEGIN {printf "%.1f", total / cards}')

    # # 内存使用
    # mem_use=$(free -h | grep Mem | awk '{print $3}')

    # # 更新最大值记录
    # MAX_AICORE=$(awk -v avg="$aicore_avg" -v max="$MAX_AICORE" 'BEGIN {print (avg > max) ? avg : max}')
    # MAX_HBM=$(awk -v avg="$hbm_avg" -v max="$MAX_HBM" 'BEGIN {print (avg > max) ? avg : max}')
    # MAX_BANDWIDTH=$(awk -v avg="$bandwidth_avg" -v max="$MAX_BANDWIDTH" 'BEGIN {print (avg > max) ? avg : max}')

    # 格式化输出
    # timestamp=$(date "+%H:%M:%S")
    # printf "%s   %7.1f%%  %4.1f%%  %7.1f%%  %4.1f%%   %7.1f%%  %4.1f%%   %6.5s \n" \
    #     "$timestamp" \
    #     $aicore_avg $MAX_AICORE \
    #     $hbm_avg $MAX_HBM \
    #     $bandwidth_avg $MAX_BANDWIDTH \
    #     $mem_use

    sleep $INTERVAL
done