# 数值分析hw1



```python
import math

def get_significant_figure(ref, est):
    """计算实数估计值 est 相对于实数参考值 ref 的有效数字位数
         注:应支持负数情况。【可选】支持 1.5e-3 科学计数法形式的输入
    Args:
        ref (str): 实数参考值的字符串形式
        est (str): 实数估计值的字符串形式
    Returns:
        n (int): 有效数字位数
    """
    
    def parse_to_parts(s):
        """将数字字符串解析为符号、整数部分、小数部分、指数"""
        s = s.strip()
        sign = 1
        if s.startswith('-'):# 处理负数
            sign = -1
            s = s[1:]
        elif s.startswith('+'):
            s = s[1:]
        
        # 处理科学计数法
        exponent = 0
        if 'e' in s.lower(): 
            parts = s.lower().split('e')
            s = parts[0]
            exponent = int(parts[1])
        
        # 分离整数和小数部分
        if '.' in s:
            int_part, dec_part = s.split('.')
        else:
            int_part, dec_part = s, ''
        
        return sign, int_part, dec_part, exponent
    
    def normalize_number(sign, int_part, dec_part, exponent):
        """将数字标准化为 ±d.ddd... × 10^m 的形式，返回符号、所有数字、数量级m"""
        # 合并所有数字
        all_digits = int_part + dec_part
        
        # 找到第一个非零数字
        first_nonzero = -1
        for i, c in enumerate(all_digits):
            if c != '0':
                first_nonzero = i
                break
        
        if first_nonzero == -1:
            # 全是0
            return sign, '0', 0
        
        # 提取有效数字
        sig_digits = all_digits[first_nonzero:]
        
        # 计算数量级 m
        original_decimal_pos = len(int_part)
        # 第一个非零数字的位置
        first_nonzero_pos = first_nonzero
        # 数量级 = 小数点位置 - 第一个非零数字位置 - 1 + exponent
        m = original_decimal_pos - first_nonzero_pos - 1 + exponent
        
        return sign, sig_digits, m
    
    # 解析两个数
    ref_sign, ref_int, ref_dec, ref_exp = parse_to_parts(ref)
    est_sign, est_int, est_dec, est_exp = parse_to_parts(est)
    
    if ref_sign * est_sign < 0:
        return 0
    
    # 标准化
    ref_sign, ref_digits, ref_m = normalize_number(ref_sign, ref_int, ref_dec, ref_exp)
    est_sign, est_digits, est_m = normalize_number(est_sign, est_int, est_dec, est_exp)
    
    # 如果ref为0
    if ref_digits == '0':
        return 0 if est_digits != '0' else 1
    
    # 如果est为0但ref不为0
    if est_digits == '0':
        return 0
    
    # = 时，有效数字位数为ref的有效位数
    if ref_digits == est_digits and ref_m == est_m:
        return len(ref_digits)
    
    # 计算ref的最大有效位数
    max_n = len(ref_digits)
    
    min_m = min(ref_m, est_m)
    
    ref_scale = len(ref_digits) - 1 - ref_m
    est_scale = len(est_digits) - 1 - est_m
    
    # 统一缩放到同一个基准
    max_scale = max(ref_scale, est_scale)
    
    # 扩展数字字符串
    ref_extended = ref_digits + '0' * (max_scale - ref_scale)
    est_extended = est_digits + '0' * (max_scale - est_scale)
    
    # 对齐长度（补齐较短的）
    max_len = max(len(ref_extended), len(est_extended))
    ref_extended = ref_extended.ljust(max_len, '0')
    est_extended = est_extended.ljust(max_len, '0')
    
    # 转换为整数计算绝对误差
    ref_int_val = int(ref_extended)
    est_int_val = int(est_extended)
    abs_error_int = abs(ref_int_val - est_int_val)
    
    # 如果误差为0（再次确认）
    if abs_error_int == 0:
        return max_n
    
    # 从max_n开始向下检查，找到满足误差限的最大n
    # 误差限公式: 0.5 × 10^(ref_m - n + 1)
    for n in range(max_n, 0, -1):
        # 计算误差限，转换为整数形式
        # error_limit = 0.5 × 10^(ref_m - n + 1)
        # 在我们的整数表示下，需要乘以 10^max_scale
        # error_limit_int = 0.5 × 10^(ref_m - n + 1 + max_scale)
        #                 = 5 × 10^(ref_m - n + max_scale)
        error_limit_exp = ref_m - n + max_scale
        
        # 处理负指数的情况
        if error_limit_exp >= 0:
            error_limit_int = 5 * (10 ** error_limit_exp)
        else:
            if abs_error_int == 0:
                return n
            # 精确比较：abs_error_int <= 5 * 10^error_limit_exp
            # 即：abs_error_int * 10^(-error_limit_exp) <= 5
            # 即：abs_error_int * 10^(abs(error_limit_exp)) <= 5
            shifted_error = abs_error_int * (10 ** abs(error_limit_exp))
            if shifted_error <= 5:
                return n
            continue
        
        # 比较实际误差和误差限
        if abs_error_int <= error_limit_int:
            return n
    
    return 0


if __name__ == "__main__":
    result1 = get_significant_figure("-3.1415926", "-3.1415")
    print(f"测试1: {result1}, 期望: 4")
    assert result1 == 4, f"测试1失败: 得到{result1}, 期望4"
    
    result2 = get_significant_figure("2.2530", "2.3000")
    print(f"测试2: {result2}, 期望: 2")
    assert result2 == 2, f"测试2失败: 得到{result2}, 期望2"
    
    result3 = get_significant_figure("-3.1415926", "-3.1416")
    print(f"测试3: {result3}, 期望: 5")
    assert result3 == 5, f"测试3失败: 得到{result3}, 期望5"
    
    result4 = get_significant_figure("3.14", "3.1416")
    print(f"测试4: {result4}, 期望: 3")
    assert result4 == 3, f"测试4失败: 得到{result4}, 期望3"
    
    result6 = get_significant_figure("0", "0.0")
    print(f"测试6: {result6}, 期望: 1")
    assert result6 == 1, f"测试6失败: 得到{result6}, 期望1"
    
    result7 = get_significant_figure("123", "-123")
    print(f"测试7: {result7}, 期望: 0")
    assert result7 == 0, f"测试7失败: 得到{result7}, 期望0"
    
    # 科学计数法测试
    result8 = get_significant_figure("1.5e-3", "1.4999e-3")
    print(f"测试8 (科学计数法): {result8}")
    
    # 高精度测试（100位小数）
    ref_100 = "3." + "1" * 100
    est_100 = "3." + "1" * 99 + "2"
    result9 = get_significant_figure(ref_100, est_100)
    print(f"测试9 (100位小数): {result9}, 期望: 100")
    
    print("\n所有测试通过!")


       # --- 新增测试 ---
    print("\n--- 新增测试 ---")
    
    # 测试10: 科学计数法与常规表示法完全相等
    # 1.23e4 等于 12300。ref有5位有效数字，est完全匹配，所以应为5
    result10 = get_significant_figure("12300", "1.2300e4")
    print(f"测试10 (格式不同但值相等): {result10}, 期望: 5")
    assert result10 == 5, f"测试10失败: 得到{result10}, 期望5"

    # 测试11: 处理小数前的零
    # ref="0.0054", est="0.005" | 误差0.0004 | 2位有效数字的误差限是0.0005
    result11 = get_significant_figure("0.0054", "0.005")
    print(f"测试11 (小数前的零): {result11}, 期望: 1")
    assert result11 == 1, f"测试11失败: 得到{result11}, 期望1"
    
    # 测试12: 四舍五入的边界情况（等于0.5倍误差）
    # ref="1.2", est="1.15" | 误差0.05 | 2位有效数字的误差限是 0.5*10^(0-2+1)=0.05。误差<=误差限，所以通过
    result12 = get_significant_figure("1.2", "1.15")
    print(f"测试12 (精确边界等于): {result12}, 期望: 2")
    assert result12 == 2, f"测试12失败: 得到{result12}, 期望2"

    # 测试13: 四舍五入的边界情况（大于0.5倍误差）
    # ref="1.2", est="1.149" | 误差0.051 | 2位有效数字的误差限是0.05。误差>误差限，失败。降到1位有效数字通过。
    result13 = get_significant_figure("1.2", "1.149")
    print(f"测试13 (精确边界大于): {result13}, 期望: 1")
    assert result13 == 1, f"测试13失败: 得到{result13}, 期望1"
    
    # 测试14: 末尾零的重要性
    # ref="5.00", est="4.99" | 误差0.01 | 3位有效数字的误差限是0.005(失败)。2位有效数字的误差限是0.05(通过)。
    result14 = get_significant_figure("5.00", "4.99")
    print(f"测试14 (末尾零的重要性): {result14}, 期望: 2")
    assert result14 == 2, f"测试14失败: 得到{result14}, 期望2"
    
    # 测试16: ref为0，est不为0
    result16 = get_significant_figure("0.0", "0.00001")
    print(f"测试16 (ref为0): {result16}, 期望: 0")
    assert result16 == 0, f"测试16失败: 得到{result16}, 期望0"

    # 测试17: est为0, ref不为0
    result17 = get_significant_figure("100", "0")
    print(f"测试17 (est为0): {result17}, 期望: 0")
    assert result17 == 0, f"测试17失败: 得到{result17}, 期望0"
    
    # 测试18: 负数的边界情况
    # ref="-2.5", est="-2.55" | 误差0.05 | 2位有效数字的误差限是0.05。通过
    result18 = get_significant_figure("-2.5", "-2.55")
    print(f"测试18 (负数边界): {result18}, 期望: 2")
    assert result18 == 2, f"测试18失败: 得到{result18}, 期望2"

    # 测试19: 高精度测试
    ref_high = "3." + "1" * 50
    est_high = "3." + "1" * 49 + "2"
    # 误差在第50位小数，满足50位有效数字的要求
    result19 = get_significant_figure(ref_high, est_high)
    print(f"测试19 (50位小数): {result19}, 期望: 50")
    assert result19 == 50, f"测试19失败: 得到{result19}, 期望50"
```
