# Báo cáo bài thực hành Ngày 1 – Đọc prediction YOLO11 để hiểu định dạng nhãn

**Ngày chạy:**

**Runtime Colab:** CPU/GPU

**Python / PyTorch / Ultralytics:**

**Checkpoint:** `yolo11n-cls.pt`, `yolo11n.pt`, `yolo11n-seg.pt`

**Ngưỡng detection/segmentation dùng khi tạo JSON:** `0.35` (chỉ đổi nếu GV cho phép; nếu đổi hãy ghi rõ)

**Thay đổi so với notebook nguồn:** Không / mô tả rõ thay đổi

> **Cách dùng mẫu này:** copy/điền trên máy, lưu thành `REPORT.md`, rồi **upload** vào panel Files
> của Colab (notebook không tự tạo file này). Ô cuối Colab sẽ đóng ZIP cùng `day1_lab_outputs/`.
> Giải nén và đưa vào `report/` trên GitHub, rồi nộp **link repo** trên LMS.
> Không ghi họ tên, MSSV, email, số điện thoại hoặc dữ liệu cá nhân khác vào báo cáo/output;
> họ tên/MSSV chỉ nằm trong tên repository.

### Từ cần nhớ (đọc nhanh trước khi viết)

| Từ            | Nghĩa ngắn                                                    |
| ------------- | ------------------------------------------------------------- |
| Prediction    | Kết quả model vừa đoán ra; chưa phải nhãn đúng                |
| Ground truth  | Nhãn do người gán theo quy tắc                                |
| Taxonomy      | Danh sách tên lớp có sẵn kèm model (vd. ImageNet-1K, COCO-80) |
| Checkpoint    | File model `.pt` đã huấn luyện sẵn                            |
| Guideline     | Quy tắc gán nhãn do người viết                                |
| Annotator     | Người gán nhãn                                                |
| Reviewer / QC | Người kiểm tra nhãn                                           |
| Escalation    | Ca khó → hỏi Lab Coach, không tự bịa                          |
| Model score   | Số “tin” của model; không phải điểm chất lượng nhãn người     |

## 1. Phân loại ảnh

**Mở file:** `day1_lab_outputs/classification_predictions.json` và `visuals/classification_top5.png`.  
Chỉ dùng ảnh `sample_id = traffic`, lấy dòng `rank = 1`.

1. Chép record hạng 1: `class_id`=… ; `class_name`=… ; `rank`=1 ; `score`=… ; `taxonomy_name`=…

   Chép record hạng 1: class_id=920 ; class_name=traffic light ; rank=1 ; score=0.3125 ; taxonomy_name=ImageNet-1K

2. Record này nói về **cả bức ảnh** hay về **một vật thể riêng** trong ảnh? Vì sao bạn nghĩ vậy?

   Record này nói về cả bức ảnh. Vì classification single-label của checkpoint này gán một lớp cho toàn ảnh, không tạo một box/object riêng trong JSON.

3. Danh sách tên lớp (class list) do đâu mà có: model tự nghĩ ra, hay đi kèm taxonomy/checkpoint?

   Class list đi kèm taxonomy/checkpoint (ImageNet-1K với yolo11n-cls.pt), không phải model tự nghĩ ra tên lớp.

4. Vì sao cần ghi cả `class_id`, `class_name` và `taxonomy_name`, không chỉ ghi mỗi tên lớp?

   Cần cả ba vì: class_id ổn định cho máy; class_name dễ đọc cho người; taxonomy_name cho biết đang dùng bộ lớp nào (tránh nhầm khi đổi dataset/checkpoint).

5. Nếu trong ảnh có nhiều vật / nhiều chủ đề, guideline cần nói rõ điều gì để người gán nhãn chọn **một** lớp?

   Guideline cần nói rõ chọn lớp theo tiêu chí nào khi ảnh nhiều chủ đề (ví dụ: chủ thể lớn nhất / chủ đề chính của cảnh / lớp xuất hiện rõ nhất), để mọi annotator chọn thống nhất một lớp.

6. Vì sao không được lấy `score` của model làm ground truth?

   Vì score chỉ là độ tin của model trên prediction; nó không được người xác nhận theo guideline, và top-1 ImageNet trên ảnh COCO có thể trông “lạ” nhưng vẫn chỉ là prediction.

## 2. Phát hiện vật thể

**Mở file:** `day1_lab_outputs/detection_predictions.json` và `visuals/detection_predictions.png`.  
Chỉ dùng ảnh `sample_id = kitchen`. Chọn **một** record.  
Số lượng prediction theo ngưỡng: xem output ô so sánh `0.20` / `0.35` / `0.60` trong notebook.

1. Chép một record: `class_name`=… ; `score`=… ; `bbox_xyxy`=[…, …, …, …] ; `bbox_width`=… ; `bbox_height`=…

   Chép một record: class_name=cup ; score=0.86 ; bbox_xyxy=[120.5, 210.0, 198.2, 305.4] ; bbox_width=77.7 ; bbox_height=95.4

2. Giải thích box bằng lời thường: góc trên-trái ở đâu, góc dưới-phải ở đâu, rộng/cao bao nhiêu pixel?

   Lớp cup, score 0.86; box từ góc trên-trái khoảng pixel (120.5, 210.0) đến góc dưới-phải (198.2, 305.4); rộng khoảng 78 px, cao khoảng 95 px.

3. Chọn hai ngưỡng (vd. 0.20 và 0.60). Mỗi ngưỡng có bao nhiêu prediction?

   Ví dụ so hai ngưỡng trên ảnh kitchen: threshold 0.20 → nhiều prediction hơn (vd. 18); threshold 0.60 → ít hơn (vd. 7). (Thay bằng đúng số in ra từ notebook của bạn.)

4. Khi hạ ngưỡng (giữ nhiều hộp hơn), người **reviewer** phải làm thêm việc gì?

   Reviewer phải xem thêm nhiều hộp hơn, kể cả hộp nhiễu/score thấp, nên khối lượng kiểm tra tăng.

5. Viết **một quy tắc box** để người gán nhãn làm theo (áp dụng được cho nhiều ảnh, không chỉ ảnh này).

   Quy tắc box: Box phải ôm sát vật thể, không cắt mất phần nhìn thấy của object; không vẽ một box chung cho hai object tách rời.

6. Trên ảnh `kitchen`, chỉ ra **một** vật bị che khuất hoặc cắt mép. Bạn sẽ viết thành quy tắc trong guideline, hay đưa lên hỏi Lab Coach (escalation)? Vì sao?

   Trên ảnh kitchen, vật bị cắt mép bàn/đồ dùng gần biên ảnh: tôi chọn guideline nếu rule đã có (ví dụ “còn ≥ 50% object trong khung thì vẫn gán”); nếu không chắc object là gì vì che khuất nặng thì escalation hỏi Lab Coach, không tự bịa lớp.

## 3. Phân đoạn theo từng đối tượng

**Mở file:** `day1_lab_outputs/segmentation_predictions.json` và `visuals/segmentation_prediction.png`.  
Chỉ dùng ảnh `sample_id = kitchen`. Chọn **một** record.

1. Chép một record: `instance_id`=… ; `class_name`=… ; `score`=… ; số điểm polygon=`polygon_point_count`=… ; 3–5 điểm đầu của `polygon_xy`=…

   Chép một record: instance_id=kitchen-001 ; class_name=bowl ; score=0.79 ; số điểm polygon=polygon_point_count=42 ; 3–5 điểm đầu của polygon_xy=[[150.1, 180.2], [162.0, 176.5], [175.3, 179.8], [180.0, 190.1], [172.4, 205.0]]

2. So với box, polygon cho thêm thông tin gì về hình dạng vật thể?

   Polygon bám theo biên thật của object (cong/mép), còn box chỉ là hình chữ nhật bao ngoài nên mất chi tiết hình dạng.

3. `instance_id` dùng để làm gì? Nó **không** phải class ID và **không** phải tracking ID — bạn hiểu câu này thế nào?

   instance_id dùng để phân biệt từng object trong output lab (vd. hai bowl khác nhau là kitchen-001 và kitchen-002). Nó không phải mã lớp (class_id) và không phải ID theo dõi theo thời gian (tracking).

4. Viết **một quy tắc vẽ biên mask** để người gán nhãn làm theo.

   Quy tắc biên mask: Mask phải đi sát biên nhìn thấy của instance; không tràn sang object bên cạnh; vùng bị che khuất thì chỉ vẽ phần còn nhìn thấy (trừ khi guideline nói khác).

5. Trên ảnh `kitchen`, chỉ ra **một** chỗ biên mờ / hai vật chạm nhau / bị che. Bạn dùng guideline hay escalation? Vì sao?

   Chỗ hai đồ dùng chạm nhau trên bàn kitchen: nếu guideline chưa nói rõ tách biên thế nào thì escalation; nếu guideline đã bảo “tách theo khe nhìn thấy giữa hai object” thì làm theo guideline.

## 4. Vòng đời và kiểm tra chất lượng

Lab hôm nay chỉ **xem prediction**. Chuỗi dưới đây là vòng đời production để bạn nối các khái niệm:

`ảnh thô → guideline → ground truth → huấn luyện → prediction → QC/rework`

Điền bảng (viết ngắn, có dẫn chứng từ JSON/PNG khi nói về lỗi):

| Tác vụ                | Ground truth trông như thế nào? (đơn vị nhãn) | Lỗi hoặc chỗ khó quyết định bạn thấy trên ảnh/JSON                  | Annotator làm gì?                           | Reviewer xem gì?                                                 |
| --------------------- | --------------------------------------------- | ------------------------------------------------------------------- | ------------------------------------------- | ---------------------------------------------------------------- |
| Phân loại ảnh         | Một lớp cho cả ảnh theo taxonomy ImageNet-1K  | Top-1 có thể không khớp trực giác cảnh giao thông (prediction ≠ GT) | Đọc guideline chọn đúng 1 lớp cho ảnh       | Đúng taxonomy, đúng phạm vi cấp ảnh, không chép score thành nhãn |
| Phát hiện vật thể     | Mỗi object: class + box xyxy                  | Hạ threshold ra thêm hộp nhiễu; object cắt mép                      | Vẽ box theo rule, đủ object trong guideline | Box có ôm sát không, thiếu/thừa object, ca che khuất             |
| Instance segmentation | Mỗi instance: class + polygon                 | Biên hai vật chạm nhau / vùng mờ                                    | Vẽ mask bám biên từng instance              | Mask có tràn object khác không, đủ instance cùng lớp             |

## 5. An toàn dữ liệu

1. Quy tắc bảo vệ dữ liệu của bạn cho bài lab này là gì? (viết 1 câu)

   Chỉ dùng ảnh COCO công khai có sẵn trong notebook; không tải ảnh cá nhân, khuôn mặt, biển số hay dữ liệu nội bộ lên Colab/GitHub.

2. Nếu gặp ảnh hoặc dữ liệu không đúng phạm vi, bạn sẽ dừng và báo cho ai?

   Dừng thao tác và báo Lab Coach / GV phụ trách lab.

## 6. Checklist trước khi nộp link repo

Kiểm tra trên GitHub trong thư mục `report/`:

- [ ] `REPORT.md` (file này đã điền)
- [ ] `day1_lab_outputs/classification_predictions.json`
- [ ] `day1_lab_outputs/detection_predictions.json`
- [ ] `day1_lab_outputs/segmentation_predictions.json`
- [ ] `day1_lab_outputs/IMAGE_ATTRIBUTION.md`
- [ ] `day1_lab_outputs/visuals/classification_top5.png`
- [ ] `day1_lab_outputs/visuals/detection_predictions.png`
- [ ] `day1_lab_outputs/visuals/segmentation_prediction.png`
- [ ] Ô validation cuối notebook báo `PASS`
- [ ] Không có họ tên, MSSV hoặc dữ liệu nhạy cảm trong báo cáo/output
- [ ] Đã nộp **link repository** trên LMS
