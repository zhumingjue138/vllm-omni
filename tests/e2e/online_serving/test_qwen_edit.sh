#!/bin/bash
set -o pipefail # 确保管道中任一命令失败即判定为失败
TOTAL=5000
LOG_FILE="requests_$(date +%Y%m%d_%H%M%S).log"
START_TIME=$(date +%s)
SUCCESS_COUNT=0
# 初始化日志
exec > >(tee -a "$LOG_FILE") 2>&1
echo "========================================"
echo "Batch Image Edit Requests Started"
echo "Total requests: $TOTAL | Concurrency: 1"
echo "Output: output_{1..$TOTAL}.png"
echo "Log file: $LOG_FILE"
echo "Start time: $(date '+%Y-%m-%d %H:%M:%S')"
echo "========================================"

# 主循环：严格顺序执行（并发=1）
for ((i=1; i<=TOTAL; i++)); do
    TIMESTAMP=$(date '+%H:%M:%S')
    REQ_START_TIME=$(date +%s.%N) # 纳秒级精度开始时间
    echo "[$TIMESTAMP] Processing request $i/$TOTAL..."
    if curl -X POST "http://127.0.0.1:8006/v1/images/edits" \
        -F "image[]=@/nvme1n1p1/z00939163/vllm-omni/tests/e2e/online_serving/cat.png" \
        -F "image[]=@/nvme1n1p1/z00939163/vllm-omni/tests/e2e/online_serving/dog.jpg" \
        -F "prompt='将第二张图中人脸/猫狗脸转换为3D卡通形象，质量要求：毛发纹理自然电影级色彩校准，注意保留原图的外貌、肤色、发型特征，仅替换第一张财神爷/猫狗财神的用绿框框住的头部区域，耳朵样式和原图的3D卡通形象一致， （帽子紧贴头顶，未贴合部分可调整帽子摆放角度达到完全贴合）， 头部与身体过渡必须自然，无拼接痕迹，无额外元素出现在头部四周，纯白背景，不要出现绿色圆圈'" \
        -F "size=2000x2496" \
        -F "output_format=jpeg" \
        -F "output_compression=98" 2>/dev/null | \
        jq -r '.data[0].b64_json' 2>/dev/null | \
        cut -d',' -f2- 2>/dev/null | \
        base64 -d > "output_${i}.png"; then

# 验证文件非空（简单校验）
        if [ -s "output_${i}.png" ]; then
            SUCCESS_COUNT=$((SUCCESS_COUNT + 1))
            REQ_END_TIME=$(date +%s.%N)
            REQ_DURATION=$(echo "$REQ_END_TIME - $REQ_START_TIME" | bc)
            rm -f "output_${i}.png"
            echo "✓ Success: output_${i}.png saved (${TIMESTAMP}) | 耗时:${REQ_DURATION}s"
        else
            REQ_END_TIME=$(date +%s.%N)
            REQ_DURATION=$(echo "$REQ_END_TIME - $REQ_START_TIME" | bc)
            rm -f "output_${i}.png"
            echo "✗ Warning: output_${i}.png is empty, deleted | 耗时:${REQ_DURATION}s"
        fi
        else
            REQ_END_TIME=$(date +%s.%N)
            REQ_DURATION=$(echo "$REQ_END_TIME - $REQ_START_TIME" | bc)
            echo "✗ Failed: Request $i encountered an error | 耗时: ${REQ_DURATION}s"
            [ -f "output_${i}.png" ] && rm -f "output_${i}.png"
        fi
done
# 汇总统计
END_TIME=$(date +%s)
DURATION=$((END_TIME - START_TIME))
AVG_DURATION=$(echo "scale=2; $DURATION / $TOTAL" | bc)
# SUCCESS_COUNT=$(ls output_*.png 2>/dev/null | wc -l)
echo ""
echo "========================================"
echo "Batch Completed!"
echo "Total attempted: $TOTAL"
echo "Successful outputs: $SUCCESS_COUNT"
echo "Failed/empty: $((TOTAL - SUCCESS_COUNT))"
echo "Total duration: ${DURATION}s"
echo "Average per request: ${AVG_DURATION}s"
echo "Log saved to: $LOG_FILE"
echo "Outputs: output_{1..${TOTAL}}.png"
echo "========================================"