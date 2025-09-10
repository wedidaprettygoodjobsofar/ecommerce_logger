from flask import Flask, request, jsonify, render_template
from utils.data_formatter import DataFormatter
import os
from datetime import datetime
import random

app = Flask(
    import_name=__name__,
    static_url_path="/static",
    static_folder="static",
    template_folder="templates",
)
# 创建日志目录
log_dir = "logs"
os.makedirs(log_dir, exist_ok=True)
data_formatter = DataFormatter(log_dir)


@app.route("/")
def home():
    """主页，显示HTML界面"""
    return render_template("index.html", random=random)


@app.route("/api/click", methods=["POST"])
def log_click():
    """记录用户点击行为"""
    try:
        data = request.get_json()

        # 验证必需字段
        required_fields = ["session_id", "item_id", "category"]
        for field in required_fields:
            if field not in data:
                return jsonify({"error": f"缺少必需字段: {field}"}), 400

        # 记录点击
        click_data = data_formatter.log_click(
            session_id=data["session_id"],
            item_id=data["item_id"],
            category=data["category"],
        )

        return jsonify({"message": "点击记录成功", "data": click_data}), 201

    except Exception as e:
        return jsonify({"error": str(e)}), 500


@app.route("/api/purchase", methods=["POST"])
def log_purchase():
    """记录用户购买行为"""
    try:
        data = request.get_json()

        # 验证必需字段
        required_fields = ["session_id", "item_id", "price", "quantity"]
        for field in required_fields:
            if field not in data:
                return jsonify({"error": f"缺少必需字段: {field}"}), 400

        # 记录购买
        purchase_data = data_formatter.log_purchase(
            session_id=data["session_id"],
            item_id=data["item_id"],
            price=data["price"],
            quantity=data["quantity"],
        )

        return jsonify({"message": "购买记录成功", "data": purchase_data}), 201

    except Exception as e:
        return jsonify({"error": str(e)}), 500


@app.route("/api/training-data", methods=["GET"])
def get_training_data():
    """获取AI训练数据"""
    try:
        clicks_df, buys_df = data_formatter.get_training_data()

        if clicks_df is None:
            return jsonify({"message": "暂无数据"}), 404

        response = {
            "clicks_count": len(clicks_df),
            "buys_count": len(buys_df) if buys_df is not None else 0,
            "clicks_sample": (
                clicks_df.head().to_dict("records") if not clicks_df.empty else []
            ),
            "buys_sample": (
                buys_df.head().to_dict("records")
                if buys_df is not None and not buys_df.empty
                else []
            ),
        }

        return jsonify(response), 200

    except Exception as e:
        return jsonify({"error": str(e)}), 500


@app.route("/api/health", methods=["GET"])
def health_check():
    """健康检查端点"""
    return (
        jsonify(
            {
                "status": "healthy",
                "timestamp": datetime.now().isoformat(),
                "log_directories": {
                    "clicks": os.path.exists("logs/clicks"),
                    "buys": os.path.exists("logs/buys"),
                },
            }
        ),
        200,
    )


@app.route("/api/stats", methods=["GET"])
def get_stats():
    """获取统计数据"""
    try:
        # 获取统计数据
        stats_data = data_formatter.get_statistics()
        return jsonify(stats_data)
    except Exception as e:
        return jsonify({"error": str(e)}), 500


if __name__ == "__main__":
    # 确保日志目录存在
    os.makedirs("logs/clicks", exist_ok=True)
    os.makedirs("logs/buys", exist_ok=True)

    app.run(debug=True, host="0.0.0.0", port=5000)
