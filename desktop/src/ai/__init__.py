# ==========================================
# 3. AI / PREDICTION LOGIC
# ==========================================
from PIL import ImageStat

class GameSeeAI:
    def predict(self, image, cpu_usage, input_count):
        """
        input_count: Number of clicks/keys in the last 10 seconds.
        """
        if image is None: return "Unknown", 0.0

        score = 0

        # --- FEATURE 1: APM (Actions) ---
        # Games usually have high APM. Movies have 0.
        if input_count > 100: score += 40    # Intense clicking/typing
        elif input_count > 20: score += 10   # Casual use
        elif input_count == 0: score -= 30   # Passive (Video?)

        # --- FEATURE 2: CPU ---
        if cpu_usage > 30: score += 20
        if cpu_usage > 60: score += 15

        # --- FEATURE 3: Visual Complexity ---
        stat = ImageStat.Stat(image)
        avg_std_dev = sum(stat.stddev) / 3

        if avg_std_dev > 50: score += 30      # Complex image
        elif avg_std_dev < 20: score -= 20    # Flat colors (Office apps)

        # --- FINAL DECISION ---
        probability = min(max(score, 0), 100)

        # Logic refinement based on combinations
        label = "APPLICATION"

        if probability > 60:
            label = "GAME"
        elif input_count > 150 and cpu_usage < 15:
            # Lots of typing but low CPU? Probably writing code or essay
            label = "TYPING/WORK"
            probability = 20
        elif input_count < 5 and cpu_usage < 10:
            label = "IDLE/READING"

        return label, probability
