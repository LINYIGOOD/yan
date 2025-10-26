# yan
resample_exp_tb_eu.py 数据预处理步骤第一步，数据重采样，将多个或单个.hdf5中数据以30s间隔进行重采样，30s内取平均，以.csv文件输出
代码使用 - - python resample_exp_tb_eu.py --input "C:/Users/CheerLin/Desktop/water_data/pv_*.hdf5" --output "C:/Users/CheerLin/Desktop/water_data/EXP_TB_EU_resampled_30s.csv" 修改对应路径


impute_missing_values_new.py 数据预处理第三步，数据缺失值填补，将数据异常值清理后的.csv文件进行填补，开关量和模拟量都为前向填充方式，由谱仪数据采集特性决定
代码使用 - - python ./water_data_process/new/impute_missing_values_new.py   --input "D:/learningtools/PyCharm 2023.1/project/forecast/water_data_process/new/EXP_TB_EU_resampled_30s_clean.csv"   --output "D:/learningtools/PyCharm 2023.1/project/forecast/water_data_process/new/result_new/EXP_TB_EU_resampled_30s_impute.csv"   --report "D:/learningtools/PyCharm 2023.1/project/forecast/water_data_process/new/result_new/EXP_TB_EU_resampled_30s_report.csv"   --mask-output "D:/learningtools/PyCharm 2023.1/project/forecast/water_data_process/new/result_new/EXP_TB_EU_resampled_30s_mask.csv"

