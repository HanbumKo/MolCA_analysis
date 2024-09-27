from PIL import Image
import os

os.makedirs("analysis_results", exist_ok=True)
os.makedirs("analysis_results/merged_maps", exist_ok=True)


def merge_images(i):
    # 여섯 개의 경로에서 이미지 불러오기
    paths = [
        # f'analysis_results/stage1_keyword_random/cross_attention_maps/attention_maps_{i}.png',
        f'analysis_results/stage1_origin/cross_attention_maps/attention_maps_{i}.png',
        # f'analysis_results/stage1_Extended/cross_attention_maps/attention_maps_{i}.png',
        # f'analysis_results/stage2_keyword_random/cross_attention_maps/attention_maps_{i}.png',
        f'analysis_results/stage2_origin/cross_attention_maps/attention_maps_{i}.png',
        # f'analysis_results/stage2_Extended/cross_attention_maps/attention_maps_{i}.png'
    ]
    
    # 이미지를 리스트에 로드
    images = [Image.open(path) for path in paths]

    # 각각의 이미지 크기가 같다고 가정, 첫 번째 이미지의 크기 확인
    width, height = images[0].size

    # 최종적으로 합쳐질 이미지의 크기 (2행 3열로 합침)
    total_width = width * 2  # 3열
    total_height = height * 1  # 2행
    merged_image = Image.new('RGB', (total_width, total_height))

    # 이미지를 행렬로 배치
    # 첫 번째 행 (A, B, C)
    merged_image.paste(images[0], (0, 0))           # A
    merged_image.paste(images[1], (width, 0))       # B
    # merged_image.paste(images[2], (2 * width, 0))   # C

    # 두 번째 행 (D, E, F)
    # merged_image.paste(images[2], (0, height))      # D
    # merged_image.paste(images[3], (width, height))  # E
    # merged_image.paste(images[5], (2 * width, height))  # F

    # 결과 이미지 저장
    merged_image.save(f'analysis_results/merged_maps/merged_attention_maps_{i}.png')

# 0부터 100까지 이미지 병합
for i in range(101):
    merge_images(i)
