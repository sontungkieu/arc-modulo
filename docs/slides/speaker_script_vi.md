# Kịch bản thuyết trình tiếng Việt

Deck: **Reproducing Test-Time Diffusion Alignment**  
Thời lượng gợi ý: 22-28 phút, trung bình 60-75 giây mỗi slide.

## Slide 1 - Reproducing Test-Time Diffusion Alignment

Trong bài trình bày này, tôi không chỉ hỏi code có chạy được hay không. Tôi muốn trả lời một câu hỏi chặt hơn: con số trong paper có được tái lập đúng metric, đúng cấu hình và đủ ổn định hay không. Bốn công trình được kiểm tra là DAS, Feynman-Kac Steering, Feynman-Kac Correctors và Null-TTA. Điểm xuyên suốt là phải tách ba mức bằng chứng: pipeline chạy được, point estimate khớp, và kết quả có ổn định qua seed hoặc run hay không.

## Slide 2 - Evidence contract

Trước khi gán PASS hay FAIL, tôi dùng bốn cổng. Thứ nhất là identity: đúng code, commit, checkpoint và cấu hình hay chưa. Thứ hai là execution: payload khoa học đã hoàn tất và lưu đủ artifact chưa. Thứ ba là numeric match: metric nào là primary, hướng tốt hơn là tăng hay giảm, tolerance đã định nghĩa trước chưa. Cuối cùng là stability: phải giữ tất cả seed và run, không chỉ chọn seed đẹp. Vì vậy DAS và FK Correctors được ghi là PASS WITH VARIANCE; FK Steering mới chỉ là PARTIAL; Null-TTA là PASS trong đúng phạm vi Table 1 SD-v1.5.

## Slide 3 - DAS: mục tiêu là phân phối nghiêng

DAS tạo phân phối mục tiêu bằng cách nghiêng phân phối pretrained với hàm reward. Trong toy GMM, reward phạt trục dọc mạnh hơn trục ngang, nhưng mục tiêu không phải dồn mọi mẫu vào điểm reward lớn nhất. Mục tiêu là toàn bộ phân phối ba mode sau khi tilt. Một lỗi quan trọng được phát hiện là notebook seed NumPy ở Python nhưng resampling nằm trong hàm Numba JIT, vốn có RNG riêng. Rescue run vì vậy khóa đồng thời Python, NumPy, Torch và Numba. Khi làm vậy, seed mặc định 42 đạt EMD 0.743, khá sát point 0.82 của paper, nhưng seed 43 lại lên 1.826.

## Slide 4 - DAS metric: EMD khác reward

Metric chính là empirical 1-Wasserstein, hay EMD với ground cost Euclidean. Nó tìm cách vận chuyển khối lượng từ tập mẫu DAS sang tập mẫu target với tổng quãng đường nhỏ nhất. EMD càng thấp thì hai phân phối càng gần về hình học và khối lượng mode. Reward chỉ là guardrail. Ví dụ, target reward khoảng âm 0.29, paper DAS âm 0.22 và reproduction seed 42 âm 0.235. Một kết quả âm 0.21 không thể tự động được gọi là tốt hơn âm 0.22, vì nó có thể đã over-optimize reward và làm sai phân phối. Trước tiên phải nhìn EMD và mode mass.

## Slide 5 - DAS canonical point và bad seed

Ở panel trái, chấm cam là EMD một lần theo đúng notebook paper; đường xanh là trung bình và độ lệch chuẩn của 100 lần subsample 500-versus-500. Seed 42 nằm trong tolerance 10 phần trăm so với paper, nhưng seed 43 cao hơn rất nhiều. Trung bình ba seed là 1.025 cộng trừ 0.704. Panel phải cho thấy reward của cả ba seed trông không quá xa paper, nhưng điều đó không cứu được seed 43 về distribution fidelity. Đây là lý do verdict phải là PASS WITH VARIANCE, không phải PASS tuyệt đối.

## Slide 6 - Mode coverage không đủ

Hình này đặt target màu xám và output DAS màu xanh. Cả ba seed đều đạt coverage 3 trên 3 theo threshold một phần trăm. Tuy nhiên seed 43 trải khối lượng sai rõ rệt, đặc biệt ở vùng nối và phần đuôi bên phải. Mode coverage chỉ hỏi mỗi mode có nhận đủ một lượng mass tối thiểu hay không; nó không hỏi tỷ lệ mass có đúng không. Mode-mass L1 của seed 43 là 0.420 và EMD là 1.826, nên hình ảnh và metric nhất quán với nhau.

## Slide 7 - DAS trên Kaggle T4x2

Kaggle xác nhận hai Tesla T4 đều visible và payload hoàn tất, nên infrastructure pass. Tuy nhiên notebook upstream chỉ chọn một GPU; T4x2 ở đây là shape cấp phát và kiểm tra portability, không phải data-parallel speedup. EMD trên Kaggle là 1.496, khác paper 0.82 và cũng khác Modal L4 0.743. Reward còn rơi xuống âm 3.230. Vì vậy phải tách verdict: hạ tầng chạy được, nhưng numeric match với paper fail và cross-backend portability với Modal cũng fail.

## Slide 8 - FK Steering: phạm vi reproduce

FK Steering dùng nhiều particle trong reverse diffusion. Sau mỗi proposal step, reward trung gian được chuyển thành potential, từ đó cập nhật weight và có thể resample theo ESS. Tôi chạy một prompt: một con dao màu nâu và một chiếc donut màu xanh, với ba seed, bốn particle, một trăm bước và lambda bằng hai. FK-max dùng reward tốt nhất đã gặp; FK-difference dùng reward tăng thêm so với trạng thái lưu. Repo model cũ trả 401 nên phải dùng public community mirror. Do identity drift này và phạm vi chỉ một prompt, không thể gọi đây là exact benchmark reproduction.

## Slide 9 - FK Steering metric

Mỗi seed và mỗi method tạo bốn particle. Central statistic là trung bình ImageReward của cả bốn ảnh, không phải chỉ lấy ảnh tốt nhất. ImageReward là scalar utility từ một reward model image-text; score có thể âm và không phải xác suất. Sau đó tôi lấy mean và sample standard deviation qua ba seed. FK-difference có mean cao nhất là 1.161 nhưng SD cũng rất lớn, 0.938. Best-of-four được giữ như diagnostic nhưng không dùng làm claim chính, vì chọn cực đại làm kết quả lạc quan hơn.

## Slide 10 - FK Steering seed variance

Đồ thị cho thấy ranking thay đổi mạnh. Ở seed 42, cả ba đường trùng nhau. Ở seed 43, FK-difference vượt xa base và FK-max. Ở seed 44, FK-max lại bằng base, trong khi FK-difference cao hơn. Ba seed trên một prompt không đủ để kết luận FK-difference tốt hơn trên GenEval. Kết quả chỉ chứng minh pipeline có thể chạy và variance cục bộ là lớn.

## Slide 11 - FK Steering qualitative result

Ba dải ảnh là FK-max, FK-difference và base tại seed 43. Nhiều ảnh có đủ knife và donut, nên đây là qualitative evidence rằng pipeline sinh ảnh hợp lệ và reward có tín hiệu. Tuy nhiên ảnh đẹp không thay thế benchmark. Scientific payload đã hoàn tất; lỗi wrapper xảy ra sau khi chín generation và metrics đã được lưu, nên đó là engineering failure sau kết quả chứ không phải scientific failure. Verdict vẫn là PARTIAL vì một prompt, ba seed và model mirror.

## Slide 12 - FK Correctors exact protocol

Khác với một bản scaled-down, thí nghiệm này chạy đủ Table A1: mười nghìn sample, một nghìn integration step, dt bằng 0.001 và năm sequential stochastic run. Có hai proposal family và ba trạng thái corrector, tạo thành sáu method; mỗi method có năm metric nên tổng cộng ba mươi ô. BDC gặp mismatch ở một call site: caller đã có cờ mô tả shape accumulator nhưng không forward nó vào sampler. Runtime copy chỉ chuyển tiếp đúng cờ có sẵn, đồng thời lưu hash của source, runtime và notebook executed. Full run mất 17.4 phút, peak khoảng 7.99 GiB và chi phí 0.367 đô la.

## Slide 13 - Năm metric của FK Correctors

W1 và W2 đo optimal transport trực tiếp trong không gian mẫu hai chiều. W2 bình phương quãng đường trước khi lấy trung bình và lấy căn, nên phạt outlier mạnh hơn W1. Cột MMD thực ra là biased MMD bình phương với tổng mười kernel RBF; trị số phụ thuộc bandwidth và không cùng đơn vị với Wasserstein. TV là xấp xỉ histogram trên lưới 200 nhân 200, nên phụ thuộc binning. Energy-W2 lại nén mỗi điểm xuống một scalar energy rồi mới so phân phối một chiều. Vì phép nén này mất thông tin, tôi tách Energy-W2 sang slide riêng.

## Slide 14 - Giải thích kỹ Energy-W2

Tên cột là Energy-W2 nhưng code gọi `pot.emd2_1d`. Hàm này dùng cost bình phương theo mặc định và code không lấy căn. Vì vậy quantity thực tế là W2 bình phương giữa hai phân phối energy, tức là giữa pushforward E thăng X và E thăng Y. Pushforward nghĩa là bỏ vị trí hai chiều, chỉ giữ danh sách scalar energy. Đơn vị của kết quả là energy bình phương; nếu nhân mọi energy với hai thì score nhân bốn.

Phản ví dụ bên phải cho thấy giới hạn quan trọng. Chọn energy bằng norm bình phương. Tập X có hai điểm trái và phải trên đường tròn đơn vị; tập Y có hai điểm trên và dưới. Tất cả đều có energy bằng một, nên hai histogram energy giống hệt nhau và Energy-W2 bằng zero. Nhưng trong không gian hai chiều, mỗi khối lượng vẫn phải di chuyển quãng đường căn hai; do đó W1 và W2 đều bằng căn hai. Kết luận: Energy-W2 thấp chỉ nói energy profile đúng, không nói geometry hai chiều đúng. :codex-annotation{index="1"}

## Slide 15 - FK Correctors: 30 trên 30 ô compatible

Heatmap biểu diễn khoảng cách có dấu giữa reproduction mean và paper mean, chia cho SD mà paper báo. Tất cả ba mươi ô nằm trong khoảng trừ hai đến cộng hai; hai mươi bốn ô còn nằm trong một SD. Hướng cải thiện của corrector so với no-FKC đồng thuận ở 19 trên 20 phép so sánh. Tuy nhiên paper mean cộng trừ hai SD chỉ là descriptive compatibility band, không phải confidence interval và cũng không phải kiểm định equivalence.

## Slide 16 - FK Correctors variance theo run

Heatmap này không nhìn mean mà hỏi run nào xa paper nhất khi tổng hợp cả năm metric. Tempered-noise BDC run zero là outlier: W2 gần 29.925 và MMD 0.176. Tempered-noise systematic run một cũng xấu, W2 20.849 và MMD 0.075. Nếu chỉ báo mean cộng trừ SD ở bảng tổng, người đọc dễ bỏ qua cấu trúc outlier này. Vì vậy verdict là PASS WITH VARIANCE: protocol và mean compatible, nhưng stability qua run chưa tốt.

## Slide 17 - Null-TTA protocol và PickScore

Null-TTA giữ nguyên trọng số diffusion và chỉ tối ưu null-text embedding tại inference. Tôi tái lập hàng Null-TTA n-max bằng 55 trên SD-v1.5, đúng target PickScore. Thiết kế đầy đủ gồm 50 prompt chính thức nhân ba seed, tức 150 paired record; mỗi cặp có baseline và optimized output. PickScore ở implementation này là cosine giữa normalized image và text embedding, không softmax và không phải xác suất. PASS yêu cầu point estimate nằm trong 10 phần trăm theo symmetric relative difference và CI bootstrap của paired gain phải hoàn toàn dương.

## Slide 18 - Ba secondary metric của Null-TTA

HPS v2 là một preference model khác PickScore, dùng để xem gain có chuyển sang proxy khác hay chỉ học mẹo của PickScore. Aesthetic là image-only: CLIP feature đi qua MLP, nên có thể đo vẻ đẹp nhưng không trực tiếp biết ảnh có đúng prompt hay không. ImageReward dùng BLIP và cross-attention image-text, score có thể âm và không nằm trong khoảng zero đến một. Ba metric này giảm rủi ro reward hacking, nhưng tất cả vẫn là learned proxy và không thay thế human evaluation.

## Slide 19 - Null-TTA Table 1 result

Bốn panel đặt paper và reproduction cạnh nhau. Với SD-v1.5 baseline, các score gần như trùng paper. Với Null-TTA, PickScore reproduction là 0.316 so với 0.315; HPS v2 là 0.294; Aesthetic là 5.471 so với 5.431; ImageReward là 0.893 so với 0.946. Symmetric difference lần lượt là 0.2, 0.1, 0.7 và 5.8 phần trăm, đều nằm trong tolerance đã định trước.

## Slide 20 - Null-TTA seed sensitivity

Ba đường nối baseline với optimized trên cùng seed. Cả seed 42, 43 và 44 đều tăng PickScore từ khoảng 0.218 lên khoảng 0.314 đến 0.317. Mean reproduction là 0.316, gần đường ngang paper 0.315. Paired gain trung bình là 0.097; hierarchical bootstrap 95 phần trăm là từ 0.089 đến 0.106, hoàn toàn trên zero. Đây là bằng chứng mạnh hơn một seed đẹp vì prompt difficulty đã được kiểm soát bằng pairing và uncertainty được tính qua cả seed lẫn prompt.

## Slide 21 - Null-TTA qualitative pairs

Hai cặp minh họa là bus và nhóm người với xe đạp. Với prompt bus, PickScore tăng từ 0.213 lên 0.313; với prompt bikes, tăng từ 0.226 lên 0.330. Hình cho thấy optimization thay đổi composition và thường làm nội dung khớp prompt rõ hơn theo scorer. Tuy nhiên chúng ta không nên chọn vài ảnh đẹp để thay cho thống kê 150 record. PASS chỉ áp dụng cho Table 1 SD-v1.5 với target PickScore; nó không tự động mở rộng sang SDXL, reward khác hay preference phổ quát của con người.

## Slide 22 - Kết luận

DAS tái lập được canonical point nhưng variance qua seed cao và Kaggle không portable về số. FK Steering chạy được pipeline và sinh artifact hợp lệ nhưng phạm vi chưa đủ cho benchmark claim. FK Correctors hoàn tất exact Table A1 với 30 trên 30 ô compatible, song có bad runs rõ rệt. Null-TTA là reproduction mạnh nhất: đủ 150 paired record, bốn scorer thật và primary gain ổn định. Thông điệp cuối cùng là không nên gọi một score là thành công chỉ vì nó trông cao hoặc thấp hơn. Claim đáng tin nhất là claim hẹp nhất vẫn sống sót sau kiểm tra identity, metric, variance và artifact.

## Ghi chú trả lời câu hỏi thường gặp

- **Vì sao không gọi DAS seed 44 EMD 0.505 là tốt hơn paper 0.82?** Vì mục tiêu là tái lập point estimate và protocol, không phải thi chọn seed tốt nhất; chọn riêng seed 44 sau khi xem kết quả là selection bias.
- **Vì sao FK Correctors 30/30 vẫn chỉ PASS WITH VARIANCE?** Vì compatibility của mean và stability của từng run là hai estimand khác nhau.
- **Vì sao Null-TTA ImageReward thấp hơn paper vẫn PASS?** Sai khác 5.8 phần trăm nằm trong tolerance 10 phần trăm đã chốt trước, trong khi primary PickScore khớp 0.2 phần trăm và paired-gain CI dương.
- **Energy-W2 có phải metric sai không?** Không. Nó đo đúng sự khớp của phân phối energy, nhưng tên cột dễ khiến người đọc tưởng nó là W2 trong không gian mẫu. Cần báo đúng semantics và đọc cùng W1/W2 2D.
