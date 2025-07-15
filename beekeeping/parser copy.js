const fs = require('fs');
const path = require('path');

const fs  = require('./fs_core');

// 변환할 JSON 파일 경로
const jsonFilePath = path.join(__dirname, 'data.json'); // 'data.json' 대신 실제 JSON 파일명 사용
const outputDir = path.join(__dirname, 'yolo_labels'); // YOLO 라벨을 저장할 디렉토리

// 출력 디렉토리 생성 (없으면)
if (!fs.existsSync(outputDir)) {
    fs.mkdirSync(outputDir);
}

fs.readFile(jsonFilePath, 'utf8', (err, data) => {
    if (err) {
        console.error('JSON 파일을 읽는 중 오류가 발생했습니다:', err);
        return;
    }

    try {
        const jsonData = JSON.parse(data);
        const imageWidth = jsonData.IMAGE.WIDTH;
        const imageHeight = jsonData.IMAGE.HEIGHT;
        const imageFileName = jsonData.IMAGE.IMAGE_FILE_NAME;

        // '알'에 대한 클래스 ID는 0으로 가정합니다.
        const classId = 0; 

        let yoloAnnotations = [];

        jsonData.ANNOTATION_INFO.forEach(annotation => {
            const xtl = annotation.XTL;
            const ytl = annotation.YTL;
            const xbr = annotation.XBR;
            const ybr = annotation.YBR;

            // 바운딩 박스의 너비와 높이 계산
            const width = xbr - xtl;
            const height = ybr - ytl;

            // 바운딩 박스의 중심 좌표 계산
            const x_center = xtl + (width / 2);
            const y_center = ytl + (height / 2);

            // 이미지 크기에 맞춰 정규화
            const x_center_normalized = x_center / imageWidth;
            const y_center_normalized = y_center / imageHeight;
            const width_normalized = width / imageWidth;
            const height_normalized = height / imageHeight;

            // YOLO 형식 문자열 생성 (소수점 6자리까지 표현)
            yoloAnnotations.push(
                `${classId} ${x_center_normalized.toFixed(6)} ${y_center_normalized.toFixed(6)} ${width_normalized.toFixed(6)} ${height_normalized.toFixed(6)}`
            );
        });

        // 결과 `.txt` 파일 경로 설정
        const outputTxtFileName = imageFileName.replace(/\.[^/.]+$/, "") + '.txt'; // 확장자 제거하고 .txt 붙이기
        const outputTxtFilePath = path.join(outputDir, outputTxtFileName);

        // 파일 쓰기
        fs.writeFile(outputTxtFilePath, yoloAnnotations.join('\n'), 'utf8', (err) => {
            if (err) {
                console.error('YOLO 형식 파일을 쓰는 중 오류가 발생했습니다:', err);
                return;
            }
            console.log(`YOLO 형식 데이터가 ${outputTxtFilePath} 파일에 성공적으로 저장되었습니다.`);
        });

    } catch (parseError) {
        console.error('JSON 데이터를 파싱하는 중 오류가 발생했습니다:', parseError);
    }
});