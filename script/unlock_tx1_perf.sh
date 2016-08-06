# Taken from https://devtalk.nvidia.com/default/topic/935300/jetson-tx1/deep-learning-inference-performance-validation-on-tx1/post/4876280/#4876280

echo "Set Tegra CPUs to max freq"
echo userspace > /sys/devices/system/cpu/cpu0/cpufreq/scaling_governor
echo userspace > /sys/devices/system/cpu/cpu1/cpufreq/scaling_governor
echo userspace > /sys/devices/system/cpu/cpu2/cpufreq/scaling_governor
echo userspace > /sys/devices/system/cpu/cpu3/cpufreq/scaling_governor
cat /sys/devices/system/cpu/cpu0/cpufreq/scaling_max_freq > /sys/devices/system/cpu/cpu0/cpufreq/scaling_min_freq
cat /sys/devices/system/cpu/cpu1/cpufreq/scaling_max_freq > /sys/devices/system/cpu/cpu1/cpufreq/scaling_min_freq
cat /sys/devices/system/cpu/cpu2/cpufreq/scaling_max_freq > /sys/devices/system/cpu/cpu2/cpufreq/scaling_min_freq
cat /sys/devices/system/cpu/cpu3/cpufreq/scaling_max_freq > /sys/devices/system/cpu/cpu3/cpufreq/scaling_min_freq
echo "Disable Tegra CPUs quite and set current gov to runnable"
echo 0 > /sys/devices/system/cpu/cpuquiet/tegra_cpuquiet/enable
echo runnable > /sys/devices/system/cpu/cpuquiet/current_governor
echo "Set Max GPU rate"
echo 844800000 > /sys/kernel/debug/clock/override.gbus/rate
echo 1 > /sys/kernel/debug/clock/override.gbus/state
# burst EMC freq to top
echo "Set Max EMC rate"
echo 1 > /sys/kernel/debug/clock/override.emc/state
cat /sys/kernel/debug/clock/emc/max > /sys/kernel/debug/clock/override.emc/rate