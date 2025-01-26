from rembg import remove
import PIL.Image

# 讀取圖片
input_path = 'input/snake.png'
output_path = 'snake_rembg.png'

# 讀取圖片
input_image = PIL.Image.open(input_path)

# 去背
output_image = remove(input_image)

# 保存結果（去背後的圖片會自動保存為 PNG 格式以保留透明度）
output_image.save(output_path)
