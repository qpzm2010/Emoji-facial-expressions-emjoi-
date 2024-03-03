class EmotionMapping:
    def __init__(self):
        # 定义面部表情到表情符号的映射
        self.emotion_mapping = {
            "Happy": "😄",
            "Sad": "😢",
            "Angry": "😡",
            "Surprise": "😯",
            "Neutral": "😐",
            "Disgust": "🤢",
            "Fear": "😨"
        }

    def get_emoji(self, emotion):
        # 根据面部表情获取相应的表情符号
        return self.emotion_mapping.get(emotion, "Unknown")

if __name__ == "__main__":
    # 创建 EmotionMapping 实例
    emotion_mapper = EmotionMapping()

    # 替换成你的面部表情识别结果
    recognized_emotion = "Happy"

    # 获取相应的表情符号
    emoji_symbol = emotion_mapper.get_emoji(recognized_emotion)

    # 打印结果
    print(f"The emoji for {recognized_emotion} is: {emoji_symbol}")
