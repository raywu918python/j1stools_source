import random

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.signal import argrelextrema

# 1. 模擬包含一個「近期收斂」的數據
# np.random.seed(42)
# dates = pd.date_range(start="2026-01-01", periods=150)
# wave = np.sin(np.linspace(0, 20, 150))
# # 讓後半段 (最近) 的波動越來越小，形成收斂
# shrink = np.linspace(25, 2, 150)
# close = 100 + wave * shrink
# df = pd.DataFrame({"Close": close}, index=dates)
#


class TriangleFeature:
    def find_triangle(df: np.DataFrame):
        # 2. 找出轉折高低點 (order=4)
        df["High_Pivot"] = df["Close"].iloc[argrelextrema(df["Close"].values, np.greater_equal, order=4)[0]]
        df["Low_Pivot"] = df["Close"].iloc[argrelextrema(df["Close"].values, np.less_equal, order=4)[0]]

        high_idx = np.where(df["High_Pivot"].notna())[0]
        low_idx = np.where(df["Low_Pivot"].notna())[0]

        high_vals = df["High_Pivot"].dropna().values
        low_vals = df["Low_Pivot"].dropna().values

        # 3. 從「最右側 (最新)」往回尋找最近的一個有效三角形
        found_triangle = False
        target_upper = None
        target_lower = None
        start_x, end_x = 0, 0

        # 雙層倒序迴圈：從最後的高低點往回找
        for j in range(len(high_idx) - 1, 0, -1):
            for i in range(j - 1, -1, -1):
                x_h1, x_h2 = high_idx[i], high_idx[j]
                y_h1, y_h2 = high_vals[i], high_vals[j]

                m_high = (y_h2 - y_h1) / (x_h2 - x_h1)
                if m_high >= 0:  # 壓力線必須向下
                    continue

                for l in range(len(low_idx) - 1, 0, -1):
                    for k in range(l - 1, -1, -1):
                        x_l1, x_l2 = low_idx[k], low_idx[l]
                        y_l1, y_l2 = low_vals[k], low_vals[l]

                        m_low = (y_l2 - y_l1) / (x_l2 - x_l1)
                        if m_low <= 0:  # 支撐線必須向上
                            continue

                        # 幾何條件過濾
                        start_x = max(x_h1, x_l1)
                        end_x = min(x_h2, x_l2)

                        if end_x <= start_x:
                            continue

                        b_high = y_h1 - m_high * x_h1
                        b_low = y_l1 - m_low * x_l1

                        y_start_high = m_high * start_x + b_high
                        y_start_low = m_low * start_x + b_low

                        if y_start_high <= y_start_low:
                            continue

                        # 🔍 額外條件：這個三角形的右端點，必須離「現在」很近 (例如在最後 30 根 K 棒內)
                        # 這樣才能確保是「最近」且「有效」的形態
                        if (len(df) - end_x) > 30:
                            continue

                        # 賓果！找到了
                        found_triangle = True
                        m_h_final, b_h_final = m_high, b_high
                        m_l_final, b_l_final = m_low, b_low
                        final_points = (x_h1, x_h2, x_l1, x_l2)
                        break
                    if found_triangle:
                        break
                if found_triangle:
                    break
            if found_triangle:
                break

        return found_triangle

        # 4. 繪圖
        def showchat():
            plt.figure(figsize=(15, 6))
            plt.plot(df.index, df["Close"], label="Close Price", color="gray")

            if found_triangle:
                x_h1, x_h2, x_l1, x_l2 = final_points

                # 延伸畫線到圖表最右側 (現在)
                plot_x = np.arange(min(x_h1, x_l1), len(df))
                upper_line = m_h_final * plot_x + b_h_final
                lower_line = m_l_final * plot_x + b_l_final

                plt.plot(
                    df.index[plot_x],
                    upper_line,
                    "--",
                    color="red",
                    label="Recent Resistance",
                    linewidth=2,
                )
                plt.plot(
                    df.index[plot_x],
                    lower_line,
                    "--",
                    color="green",
                    label="Recent Support",
                    linewidth=2,
                )

                # 標出決定這兩條線的關鍵 4 個轉折點
                plt.scatter(
                    df.index[[x_h1, x_h2]],
                    [high_vals[i], high_vals[j]],
                    color="red",
                    s=100,
                    zorder=3,
                )
                plt.scatter(
                    df.index[[x_l1, x_l2]],
                    [low_vals[k], low_vals[l]],
                    color="green",
                    s=100,
                    zorder=3,
                )

                print(f"🎯 成功鎖定最近一個三角收斂！(跨度從第 {min(x_h1, x_l1)} 根到第 {max(x_h2, x_l2)} 根 K 棒)")
            else:
                print("❌ 近期內沒有發現符合條件的三角收斂形態。")

            plt.title("Focus on the Most Recent Triangle Convergence")
            plt.legend()
            plt.show()

    def test():
        # list_random_stock_id = random.sample(get_file_list(), 3)

        # for i in list_random_stock_id:
        #     stock_id = i
        #     df = stock(stock_id)
        #     df["Close"] = df["close"]
        #     df = df[-30:]
        #     print(f"{stock_id} = {TriangleFeature.find_triangle(df)}")
        pass
