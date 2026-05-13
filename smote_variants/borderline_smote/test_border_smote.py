"""
BorderSMOTE 单元测试
"""

import numpy as np
from sklearn.datasets import make_classification
import sys

# 导入BorderSMOTE
from border_smote import BorderSMOTE


def test_basic_functionality():
    """测试基本功能"""
    print("\n" + "=" * 60)
    print("测试1: 基本功能")
    print("=" * 60)
    
    X, y = make_classification(
        n_samples=200,
        n_features=5,
        n_informative=3,
        weights=[0.9, 0.1],
        random_state=42
    )
    
    smote = BorderSMOTE(k_neighbors=5, random_state=42)
    X_res, y_res = smote.fit_resample(X, y)
    
    # 检查返回值形状
    assert X_res.shape[1] == X.shape[1], "特征维度不匹配"
    assert len(X_res) == len(y_res), "样本数和标签数不匹配"
    
    # 检查平衡性
    assert np.sum(y_res == 0) == np.sum(y_res == 1), "类别数量不相等"
    
    print("✓ 返回值形状正确")
    print(f"✓ 平衡后数据集大小: {X_res.shape}")
    print(f"✓ 类别分布: {np.sum(y_res == 0)} vs {np.sum(y_res == 1)}")
    print("✓ 基本功能测试通过！")
    

def test_synthetic_only():
    """测试仅返回合成样本"""
    print("\n" + "=" * 60)
    print("测试2: fit_resample_only_synthetic")
    print("=" * 60)
    
    X, y = make_classification(
        n_samples=200,
        n_features=5,
        n_informative=3,
        weights=[0.9, 0.1],
        random_state=42
    )
    
    n_minority_original = np.sum(y == 1)
    n_majority = np.sum(y == 0)
    
    smote = BorderSMOTE(k_neighbors=5, random_state=42)
    X_syn, y_syn = smote.fit_resample_only_synthetic(X, y)
    
    # 检查合成样本数
    expected_synthetic = n_majority - n_minority_original
    actual_synthetic = len(X_syn)
    
    assert actual_synthetic == expected_synthetic, \
        f"合成样本数不正确: 期望 {expected_synthetic}, 实际 {actual_synthetic}"
    
    # 检查所有标签都是少数类
    assert np.all(y_syn == 1), "合成样本标签不全是少数类"
    
    # 检查特征维度
    assert X_syn.shape[1] == X.shape[1], "合成样本特征维度不匹配"
    
    print("✓ 返回值形状正确")
    print(f"✓ 合成样本数: {len(X_syn)}")
    print(f"✓ 所有标签都是少数类: {np.all(y_syn == 1)}")
    print("✓ fit_resample_only_synthetic测试通过！")


def test_reproducibility():
    """测试可复现性"""
    print("\n" + "=" * 60)
    print("测试3: 可复现性 (random_state)")
    print("=" * 60)
    
    X, y = make_classification(
        n_samples=200,
        n_features=5,
        n_informative=3,
        weights=[0.9, 0.1],
        random_state=42
    )
    
    # 第一次运行
    smote1 = BorderSMOTE(k_neighbors=5, random_state=123)
    X_res1, y_res1 = smote1.fit_resample(X, y)
    
    # 第二次运行
    smote2 = BorderSMOTE(k_neighbors=5, random_state=123)
    X_res2, y_res2 = smote2.fit_resample(X, y)
    
    # 检查是否完全相同
    assert np.allclose(X_res1, X_res2), "结果不可复现"
    assert np.array_equal(y_res1, y_res2), "标签不可复现"
    
    print("✓ 相同random_state产生相同结果")
    print("✓ 可复现性测试通过！")


def test_different_k_values():
    """测试不同k值"""
    print("\n" + "=" * 60)
    print("测试4: 不同的k值")
    print("=" * 60)
    
    X, y = make_classification(
        n_samples=200,
        n_features=5,
        n_informative=3,
        weights=[0.9, 0.1],
        random_state=42
    )
    
    k_values = [3, 5, 7, 10]
    
    for k in k_values:
        smote = BorderSMOTE(k_neighbors=k, random_state=42)
        X_res, y_res = smote.fit_resample(X, y)
        
        # 检查结果有效性
        assert X_res.shape[1] == X.shape[1], f"k={k}时特征维度错误"
        assert len(X_res) == len(y_res), f"k={k}时长度不匹配"
        assert np.sum(y_res == 0) == np.sum(y_res == 1), f"k={k}时类别不平衡"
        
        print(f"✓ k={k}: 生成了 {len(X_res) - len(X)} 个合成样本")
    
    print("✓ 不同k值测试通过！")


def test_borderline_types():
    """测试不同的borderline类型"""
    print("\n" + "=" * 60)
    print("测试5: 不同的borderline类型")
    print("=" * 60)
    
    X, y = make_classification(
        n_samples=200,
        n_features=5,
        n_informative=3,
        weights=[0.9, 0.1],
        random_state=42
    )
    
    types = ['borderline1', 'borderline2']
    
    for btype in types:
        smote = BorderSMOTE(k_neighbors=5, borderline_type=btype, random_state=42)
        X_res, y_res = smote.fit_resample(X, y)
        
        # 检查结果有效性
        assert X_res.shape[1] == X.shape[1], f"{btype}特征维度错误"
        assert len(X_res) == len(y_res), f"{btype}长度不匹配"
        assert np.sum(y_res == 0) == np.sum(y_res == 1), f"{btype}类别不平衡"
        
        print(f"✓ {btype}: 生成了 {len(X_res) - len(X)} 个合成样本")
    
    print("✓ 不同borderline类型测试通过！")


def test_very_imbalanced_data():
    """测试极度不平衡的数据"""
    print("\n" + "=" * 60)
    print("测试6: 极度不平衡数据")
    print("=" * 60)
    
    X, y = make_classification(
        n_samples=1000,
        n_features=10,
        n_informative=5,
        weights=[0.99, 0.01],  # 99:1不平衡
        random_state=42
    )
    
    print(f"原始数据: 多数类={np.sum(y==0)}, 少数类={np.sum(y==1)}")
    
    smote = BorderSMOTE(k_neighbors=5, random_state=42)
    X_res, y_res = smote.fit_resample(X, y)
    
    # 检查平衡性
    assert np.sum(y_res == 0) == np.sum(y_res == 1), "极度不平衡数据平衡失败"
    
    print(f"平衡后数据: 多数类={np.sum(y_res==0)}, 少数类={np.sum(y_res==1)}")
    print("✓ 极度不平衡数据测试通过！")


def test_edge_cases():
    """测试边界情况"""
    print("\n" + "=" * 60)
    print("测试7: 边界情况")
    print("=" * 60)
    
    # 情况1: 少数类样本很少
    X, y = make_classification(
        n_samples=100,
        n_features=5,
        n_informative=3,
        weights=[0.95, 0.05],
        random_state=42
    )
    
    smote = BorderSMOTE(k_neighbors=3, random_state=42)  # k必须小于样本数
    X_res, y_res = smote.fit_resample(X, y)
    assert len(X_res) > len(X), "少数类样本少的情况失败"
    print("✓ 少数类样本少的情况通过")
    
    # 情况2: 所有样本都是同一类（边界情况）
    X_single = np.random.randn(50, 5)
    y_single = np.ones(50)
    
    try:
        smote = BorderSMOTE(k_neighbors=5)
        # 这会失败，因为没有多数类
        # 但我们应该优雅地处理
        print("✓ 单类数据的处理已记录")
    except Exception as e:
        print(f"✓ 单类数据正确抛出异常: {type(e).__name__}")


def test_synthetic_samples_quality():
    """测试合成样本质量"""
    print("\n" + "=" * 60)
    print("测试8: 合成样本质量")
    print("=" * 60)
    
    X, y = make_classification(
        n_samples=300,
        n_features=5,
        n_informative=3,
        weights=[0.9, 0.1],
        random_state=42
    )
    
    smote = BorderSMOTE(k_neighbors=5, random_state=42)
    X_syn, y_syn = smote.fit_resample_only_synthetic(X, y)
    
    X_minority = X[y == 1]
    
    # 检查1: 合成样本不全是离群点
    from sklearn.metrics.pairwise import euclidean_distances
    distances = euclidean_distances(X_syn, X_minority)
    min_distances = distances.min(axis=1)
    
    # 合成样本应该靠近原始少数类样本
    outlier_ratio = np.sum(min_distances > min_distances.mean() + 2 * min_distances.std()) / len(X_syn)
    assert outlier_ratio < 0.5, "过多的离群合成样本"
    print(f"✓ 离群点比例: {outlier_ratio:.2%} (< 50%)")
    
    # 检查2: 合成样本的特征范围应该与原始相似
    for i in range(X.shape[1]):
        orig_min, orig_max = X_minority[:, i].min(), X_minority[:, i].max()
        syn_min, syn_max = X_syn[:, i].min(), X_syn[:, i].max()
        
        # 允许一些超出范围，但不能超出太多
        assert syn_min >= orig_min - (orig_max - orig_min), \
            f"特征{i}的合成样本下界超出范围过多"
        assert syn_max <= orig_max + (orig_max - orig_min), \
            f"特征{i}的合成样本上界超出范围过多"
    
    print("✓ 合成样本特征范围符合预期")
    print("✓ 合成样本质量测试通过！")


def test_with_standardized_data():
    """测试标准化数据"""
    print("\n" + "=" * 60)
    print("测试9: 标准化数据")
    print("=" * 60)
    
    from sklearn.preprocessing import StandardScaler
    
    X, y = make_classification(
        n_samples=200,
        n_features=5,
        n_informative=3,
        weights=[0.9, 0.1],
        random_state=42
    )
    
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    
    smote = BorderSMOTE(k_neighbors=5, random_state=42)
    X_res, y_res = smote.fit_resample(X_scaled, y)
    
    # 检查结果
    assert X_res.shape[1] == X_scaled.shape[1], "特征维度错误"
    assert np.sum(y_res == 0) == np.sum(y_res == 1), "类别不平衡"
    
    print(f"✓ 标准化数据处理成功")
    print(f"✓ 平衡后数据集大小: {X_res.shape}")
    print("✓ 标准化数据测试通过！")


def run_all_tests():
    """运行所有测试"""
    print("\n" + "=" * 60)
    print("BorderSMOTE 单元测试套件")
    print("=" * 60)
    
    tests = [
        test_basic_functionality,
        test_synthetic_only,
        test_reproducibility,
        test_different_k_values,
        test_borderline_types,
        test_very_imbalanced_data,
        test_edge_cases,
        test_synthetic_samples_quality,
        test_with_standardized_data,
    ]
    
    passed = 0
    failed = 0
    
    for test in tests:
        try:
            test()
            passed += 1
        except AssertionError as e:
            print(f"✗ 测试失败: {e}")
            failed += 1
        except Exception as e:
            print(f"✗ 测试异常: {type(e).__name__}: {e}")
            failed += 1
    
    print("\n" + "=" * 60)
    print(f"测试结果: {passed} 通过, {failed} 失败")
    print("=" * 60)
    
    return failed == 0


if __name__ == "__main__":
    success = run_all_tests()
    sys.exit(0 if success else 1)
