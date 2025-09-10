import pandas as pd
from sklearn.preprocessing import LabelEncoder
import os
from datetime import datetime


class DataFormatter:
    def __init__(self, log_dir="logs"):
        self.log_dir = log_dir
        self.click_dir = os.path.join(log_dir, "clicks")
        self.buy_dir = os.path.join(log_dir, "buys")
        self.item_encoder = LabelEncoder()

        # 创建日志目录
        os.makedirs(self.click_dir, exist_ok=True)
        os.makedirs(self.buy_dir, exist_ok=True)

        # 初始化编码器（如果需要）
        self._init_encoder()

    def _init_encoder(self):
        """初始化标签编码器"""
        try:
            # 尝试加载现有的商品ID来拟合编码器
            all_item_ids = self._load_existing_item_ids()
            if len(all_item_ids) > 0:
                self.item_encoder.fit(all_item_ids)
        except:
            pass

    def _load_existing_item_ids(self):
        """加载所有已存在的商品ID"""
        item_ids = set()

        # 从点击日志加载
        if os.path.exists(self.click_dir):
            for filename in os.listdir(self.click_dir):
                if filename.endswith(".csv"):
                    df = pd.read_csv(os.path.join(self.click_dir, filename))
                    item_ids.update(df["item_id"].astype(str))

        # 从购买日志加载
        if os.path.exists(self.buy_dir):
            for filename in os.listdir(self.buy_dir):
                if filename.endswith(".csv"):
                    df = pd.read_csv(os.path.join(self.buy_dir, filename))
                    item_ids.update(df["item_id"].astype(str))

        return list(item_ids)

    def log_click(self, session_id, item_id, category):
        """记录点击行为"""
        timestamp = datetime.now().isoformat()

        # 创建数据记录
        click_data = {
            "session_id": session_id,
            "timestamp": timestamp,
            "item_id": str(item_id),  # 保持原始ID用于记录
            "category": category,
        }

        # 保存到CSV文件（按日期分文件）
        date_str = datetime.now().strftime("%Y%m%d")
        filename = f"clicks_{date_str}.csv"
        filepath = os.path.join(self.click_dir, filename)

        df = pd.DataFrame([click_data])
        if os.path.exists(filepath):
            df.to_csv(filepath, mode="a", header=False, index=False)
        else:
            df.to_csv(filepath, index=False)

        return click_data

    def log_purchase(self, session_id, item_id, price, quantity):
        """记录购买行为"""
        timestamp = datetime.now().isoformat()

        # 创建数据记录
        purchase_data = {
            "session_id": session_id,
            "timestamp": timestamp,
            "item_id": str(item_id),  # 保持原始ID用于记录
            "price": float(price),
            "quantity": int(quantity),
        }

        # 保存到CSV文件（按日期分文件）
        date_str = datetime.now().strftime("%Y%m%d")
        filename = f"buys_{date_str}.csv"
        filepath = os.path.join(self.buy_dir, filename)

        df = pd.DataFrame([purchase_data])
        if os.path.exists(filepath):
            df.to_csv(filepath, mode="a", header=False, index=False)
        else:
            df.to_csv(filepath, index=False)

        return purchase_data

    def get_training_data(self):
        """为AI准备训练数据（模仿你的AI源码格式）"""
        # 加载所有点击数据
        click_files = [
            os.path.join(self.click_dir, f)
            for f in os.listdir(self.click_dir)
            if f.startswith("clicks_") and f.endswith(".csv")
        ]

        if not click_files:
            return None, None

        # 合并所有点击数据
        click_dfs = []
        for file in click_files:
            try:
                df = pd.read_csv(file, low_memory=False)
                click_dfs.append(df)
            except:
                continue

        if not click_dfs:
            return None, None

        clicks_df = pd.concat(click_dfs, ignore_index=True)

        # 加载所有购买数据
        buy_files = [
            os.path.join(self.buy_dir, f)
            for f in os.listdir(self.buy_dir)
            if f.startswith("buys_") and f.endswith(".csv")
        ]

        buy_dfs = []
        for file in buy_files:
            try:
                df = pd.read_csv(file, low_memory=False)
                buy_dfs.append(df)
            except:
                continue

        buys_df = pd.concat(buy_dfs, ignore_index=True) if buy_dfs else pd.DataFrame()

        # 应用标签编码（模仿你的AI处理方式）
        if not clicks_df.empty:
            # 合并所有商品ID进行编码
            all_item_ids = pd.concat(
                [
                    clicks_df["item_id"],
                    buys_df["item_id"] if not buys_df.empty else pd.Series(),
                ]
            )
            self.item_encoder.fit(all_item_ids.astype(str))

            clicks_df["item_id"] = self.item_encoder.transform(
                clicks_df["item_id"].astype(str)
            )

        if not buys_df.empty:
            buys_df["item_id"] = self.item_encoder.transform(
                buys_df["item_id"].astype(str)
            )

        return clicks_df, buys_df

    def get_statistics(self):
        """获取统计数据，用于管理员界面展示"""
        try:
            # 获取最近7天的日期
            today = datetime.now()
            dates = [(today - timedelta(days=i)).strftime('%Y%m%d') for i in range(7)]
            display_dates = [(today - timedelta(days=i)).strftime('%m/%d') for i in range(7)][::-1]

            # 初始化统计数据
            total_clicks = 0
            total_purchases = 0
            total_revenue = 0.0
            daily_data = []
            category_counts = {}
            recent_records = []

            # 按日期初始化每日数据
            for date in display_dates:
                daily_data.append({'date': date, 'clicks': 0, 'purchases': 0})

            # 处理每个日期的文件
            for i, date in enumerate(dates):
                # 处理点击数据
                click_file = os.path.join(self.log_dir, f'clicks_{date}.csv')
                if os.path.exists(click_file):
                    try:
                        click_df = pd.read_csv(click_file)
                        click_count = len(click_df)
                        total_clicks += click_count
                        
                        # 更新每日点击数据
                        daily_data[6 - i]['clicks'] = click_count
                        
                        # 统计分类数据
                        if 'category' in click_df.columns:
                            category_data = click_df['category'].value_counts().to_dict()
                            for cat, count in category_data.items():
                                if cat not in category_counts:
                                    category_counts[cat] = 0
                                category_counts[cat] += count
                        
                        # 获取最近的点击记录
                        if len(recent_records) < 10:
                            # 按时间戳排序，获取最新的记录
                            click_df['timestamp'] = pd.to_datetime(click_df['timestamp'])
                            recent_clicks = click_df.nlargest(10 - len(recent_records), 'timestamp')
                            for _, row in recent_clicks.iterrows():
                                recent_records.append({
                                    'time': row['timestamp'].strftime('%H:%M:%S'),
                                    'user_id': row['session_id'],
                                    'item_id': row['item_id'],
                                    'type': 'click',
                                    'details': row.get('category', '')
                                })
                    except Exception as e:
                        print(f"读取点击日志文件 {date} 时出错: {e}")
                
                # 处理购买数据
                purchase_file = os.path.join(self.log_dir, f'purchases_{date}.csv')
                if os.path.exists(purchase_file):
                    try:
                        purchase_df = pd.read_csv(purchase_file)
                        purchase_count = len(purchase_df)
                        total_purchases += purchase_count
                        
                        # 计算销售额
                        if 'price' in purchase_df.columns and 'quantity' in purchase_df.columns:
                            revenue = (purchase_df['price'] * purchase_df['quantity']).sum()
                            total_revenue += revenue
                        
                        # 更新每日购买数据
                        daily_data[6 - i]['purchases'] = purchase_count
                        
                        # 获取最近的购买记录
                        if len(recent_records) < 10:
                            # 按时间戳排序，获取最新的记录
                            purchase_df['timestamp'] = pd.to_datetime(purchase_df['timestamp'])
                            recent_purchases = purchase_df.nlargest(10 - len(recent_records), 'timestamp')
                            for _, row in recent_purchases.iterrows():
                                details = f"¥{row['price']:.2f} x {row['quantity']}"
                                recent_records.append({
                                    'time': row['timestamp'].strftime('%H:%M:%S'),
                                    'user_id': row['session_id'],
                                    'item_id': row['item_id'],
                                    'type': 'purchase',
                                    'details': details
                                })
                    except Exception as e:
                        print(f"读取购买日志文件 {date} 时出错: {e}")
            
            # 按时间戳排序最近记录
            recent_records.sort(key=lambda x: x['time'], reverse=True)
            
            # 转换分类数据为图表格式
            category_data = []
            category_names = {
                'electronics': '电子产品',
                'clothing': '服装',
                'books': '图书',
                'home': '家居用品',
                'sports': '运动户外',
                'beauty': '美妆个护',
                'food': '食品饮料',
                'toys': '玩具'
            }
            
            for category, count in category_counts.items():
                display_name = category_names.get(category, '其他')
                category_data.append({'name': display_name, 'value': count})
            
            # 准备返回数据
            result = {
                'total_clicks': total_clicks,
                'total_purchases': total_purchases,
                'total_revenue': round(total_revenue, 2),
                'daily_data': daily_data,
                'category_data': category_data,
                'recent_records': recent_records[:10]  # 限制最近记录数量
            }
            
            return result
        except Exception as e:
            print(f"获取统计数据时出错: {e}")
            # 返回默认数据以避免前端错误
            return {
                'total_clicks': 0,
                'total_purchases': 0,
                'total_revenue': 0.0,
                'daily_data': [],
                'category_data': [],
                'recent_records': []
            }
