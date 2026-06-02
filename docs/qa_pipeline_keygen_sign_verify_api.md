# Q&A Pipeline ML-DSA: KeyGen, Sign, Verify, API

Tài liệu này dùng để chuẩn bị trả lời giảng viên khi bảo vệ phần pipeline chính của dự án ML-DSA. Mỗi câu có câu trả lời ngắn để nói nhanh, phần giải thích khi bị hỏi sâu, và dẫn chứng code/test để chỉ vào repo.

## A. Tổng Quan API

### Q01. `ML_DSA` trong `api.py` có vai trò gì?

**Câu hỏi:** Tại sao lại cần class `ML_DSA` thay vì gọi trực tiếp từng hàm `KeyGen`, `Sign`, `Verify`?

**Trả lời ngắn:** `ML_DSA` là wrapper gắn một bộ tham số cụ thể với các thao tác keygen, sign, verify, giúp API cấp cao dễ dùng và tránh truyền nhầm tham số.

**Giải thích khi bị hỏi sâu:** ML-DSA có 3 mức tham số là 44, 65, 87. Nếu gọi internal function trực tiếp, mỗi lần phải truyền `params`, dễ nhầm giữa key/signature của bộ này với bộ khác. Wrapper giữ `params` cố định, expose `name`, `pk_bytes`, `sk_bytes`, `sig_bytes`, và các method `keygen`, `sign`, `verify`. Nó không giữ khóa hay trạng thái bí mật, nên chỉ là interface thuận tiện.

**Dẫn chứng code/test:** `ml_dsa/api.py` class `ML_DSA`; `test/test_pipeline.py::test_keygen_returns_correct_sizes`.

### Q02. Khác nhau giữa `ML_DSA_44`, `ML_DSA_65`, `ML_DSA_87` là gì?

**Câu hỏi:** Ba object public này khác nhau ở điểm nào?

**Trả lời ngắn:** Chúng cùng dùng một pipeline, nhưng khác bộ tham số `k`, `l`, `eta`, `tau`, `gamma1`, `gamma2`, `omega`, `lambda`, nên khác mức an toàn và kích thước key/signature.

**Giải thích khi bị hỏi sâu:** `ML_DSA_44` nhẹ hơn, key/signature ngắn hơn; `ML_DSA_87` an toàn cao hơn nhưng dữ liệu lớn hơn; `ML_DSA_65` là mức cân bằng. Implementation tạo 3 instance từ `ML_DSA_44_PARAMS`, `ML_DSA_65_PARAMS`, `ML_DSA_87_PARAMS`, còn thuật toán keygen/sign/verify dùng chung code.

**Dẫn chứng code/test:** `ml_dsa/api.py` tạo `ML_DSA_44`, `ML_DSA_65`, `ML_DSA_87`; `ml_dsa/params.py` định nghĩa tham số; `test/test_params.py`.

### Q03. Vì sao wrapper được gọi là stateless?

**Câu hỏi:** Object `ML_DSA` có lưu secret key hoặc trạng thái ký không?

**Trả lời ngắn:** Không. Object chỉ lưu bộ tham số, còn khóa, message, signature đều được truyền qua tham số hàm và trả về dưới dạng bytes.

**Giải thích khi bị hỏi sâu:** Stateless giúp object có thể tái sử dụng an toàn cho nhiều lần keygen/sign/verify. Secret key không nằm trong object nên không có nguy cơ bị reuse state bất ngờ. Trạng thái duy nhất nằm trong dữ liệu input/output của từng call.

**Dẫn chứng code/test:** `ml_dsa/api.py` dataclass chỉ có field `params`; method `sign` nhận `sk` từ ngoài.

### Q04. Public API và internal API khác nhau thế nào?

**Câu hỏi:** Khi nào dùng `keygen/sign/verify`, khi nào dùng `keygen_internal/sign_internal/verify_internal`?

**Trả lời ngắn:** Public API xử lý message gốc, context, pre-hash và randomness; internal API nhận dữ liệu đã tiền xử lý như `message_prime` hoặc seed cố định để phục vụ test/KAT.

**Giải thích khi bị hỏi sâu:** Public `sign` gọi `FormatMessagePrime`, chọn `rnd`, rồi gọi `Sign_internal`. Public `verify` kiểm tra size/context, format message, rồi gọi `Verify_internal`. Internal API sát thuật toán FIPS hơn nhưng dễ dùng sai nếu caller chưa chuẩn bị đúng input.

**Dẫn chứng code/test:** `ml_dsa/api.py` các method `sign`, `sign_internal`, `verify`, `verify_internal`; `ml_dsa/sign.py::Sign`; `ml_dsa/verify.py::Verify`.

### Q05. Tại sao public key, secret key, signature đều là `bytes`?

**Câu hỏi:** Sao không trả về list đa thức hoặc object phức tạp?

**Trả lời ngắn:** Vì output chuẩn của scheme là dạng encoded bytes để lưu trữ, truyền đi, và so sánh kích thước chính xác.

**Giải thích khi bị hỏi sâu:** Bên trong thuật toán dùng vector đa thức, nhưng API bên ngoài cần dạng ổn định. `pkEncode`, `skEncode`, `sigEncode` đóng gói dữ liệu theo bit packing. Verify cũng bắt đầu từ `pkDecode` và `sigDecode`, nên interface bytes phản ánh đúng cách scheme được dùng thực tế.

**Dẫn chứng code/test:** `ml_dsa/encoding.py` các hàm encode/decode; `test/test_encoding.py`; `test/test_pipeline.py::test_keygen_returns_correct_sizes`.

### Q06. Vì sao API expose `pk_bytes`, `sk_bytes`, `sig_bytes`?

**Câu hỏi:** Các property kích thước dùng để làm gì?

**Trả lời ngắn:** Chúng giúp caller biết chính xác độ dài dữ liệu hợp lệ và giúp UI/test validate input trước khi verify.

**Giải thích khi bị hỏi sâu:** Kích thước phụ thuộc bộ tham số, nên không nên hard-code bên ngoài. Ví dụ cùng là signature nhưng ML-DSA-44, 65, 87 có độ dài khác nhau. Verify sẽ reject dữ liệu sai size để tránh decode sai.

**Dẫn chứng code/test:** `ml_dsa/api.py` properties; `ml_dsa/params.py` properties byte-size; `test/test_pipeline.py::test_verify_rejects_wrong_size_inputs`.

### Q07. API xử lý lỗi theo nguyên tắc nào?

**Câu hỏi:** Vì sao `sign` có thể raise lỗi, còn `verify` thường trả `False`?

**Trả lời ngắn:** `sign` là thao tác phía chủ khóa nên lỗi input nên được báo rõ; `verify` là thao tác kiểm tra dữ liệu bên ngoài nên trả `False` cho input sai để có hành vi xác thực đơn giản và an toàn hơn.

**Giải thích khi bị hỏi sâu:** `verify` kiểm tra context, size `pk/sig`, và bắt `ValueError` khi format message; nếu có vấn đề thì trả `False`. Điều này tránh để caller phải phân biệt nhiều exception khi dữ liệu không đáng tin. `sign` với `pre_hash` sai thì raise `ValueError` để báo cấu hình ký sai.

**Dẫn chứng code/test:** `ml_dsa/verify.py::Verify`; `test/test_pipeline.py::test_invalid_prehash_name_rejected_without_changing_pure_api`.

## B. KeyGen Pipeline

### Q08. Input `xi` trong KeyGen là gì?

**Câu hỏi:** `xi` có vai trò gì trong quá trình sinh khóa?

**Trả lời ngắn:** `xi` là seed 32 byte đầu vào cho KeyGen; nếu không truyền thì lấy ngẫu nhiên từ `os.urandom`.

**Giải thích khi bị hỏi sâu:** Với test và demo, truyền `xi` giúp kết quả deterministic để so sánh. Trong dùng thật, không truyền `xi` để hệ điều hành cấp entropy. `KeyGen_internal` bắt buộc `xi` dài đúng 32 byte.

**Dẫn chứng code/test:** `ml_dsa/keygen.py::KeyGen`, `KeyGen_internal`; `test/test_pipeline.py::test_keygen_deterministic_with_xi`.

### Q09. Vì sao seed được nối thêm `(k, l)` trước khi hash?

**Câu hỏi:** Tại sao code dùng `xi + bytes([params.k, params.l])`?

**Trả lời ngắn:** Để domain separation giữa các bộ tham số, tránh cùng seed tạo ra chuỗi trung gian giống nhau cho ML-DSA-44/65/87.

**Giải thích khi bị hỏi sâu:** Ba scheme có cùng thuật toán nhưng khác kích thước ma trận/vector. Nếu chỉ dùng cùng `xi`, các phần đầu của luồng hash có thể tương quan không cần thiết. Nối `(k,l)` làm seed đầu vào khác nhau theo parameter set.

**Dẫn chứng code/test:** `ml_dsa/keygen.py::KeyGen_internal`; `test/test_pipeline.py::test_signing_with_all_three_schemes_using_same_seed_yields_distinct_keys`.

### Q10. `rho`, `rho_prime`, `K` được sinh như thế nào?

**Câu hỏi:** Sau khi hash seed, code tách thành những phần nào?

**Trả lời ngắn:** Code dùng `HShake256(..., 128)` rồi tách `rho` 32 byte, `rho_prime` 64 byte, và `K` 32 byte.

**Giải thích khi bị hỏi sâu:** `rho` dùng để expand ma trận công khai `A`; `rho_prime` dùng để sinh secret vectors `s1`, `s2`; `K` là secret seed nằm trong secret key và được dùng khi ký để sinh `rho_pp`.

**Dẫn chứng code/test:** `ml_dsa/keygen.py::KeyGen_internal`; `ml_dsa/sign.py::Sign_internal`.

### Q11. `ExpandA` làm gì trong KeyGen?

**Câu hỏi:** Ma trận `A_hat` từ đâu ra?

**Trả lời ngắn:** `ExpandA(rho, params)` sinh ma trận công khai `A` trong miền NTT từ seed `rho`.

**Giải thích khi bị hỏi sâu:** Thay vì lưu toàn bộ ma trận lớn, scheme lưu seed `rho` trong public key. Ai có public key cũng tái tạo được `A`. Trong KeyGen, `A_hat` được dùng để tính `A*s1`.

**Dẫn chứng code/test:** `ml_dsa/keygen.py::KeyGen_internal`; `ml_dsa/sampling.py::ExpandA`; `test/test_sampling.py::test_expand_a_dimensions_and_determinism`.

### Q12. `ExpandS` sinh những gì?

**Câu hỏi:** `s1` và `s2` có vai trò gì?

**Trả lời ngắn:** `ExpandS(rho_prime, params)` sinh hai vector bí mật nhỏ `s1` và `s2`, dùng để tạo khóa và sau này dùng trong ký.

**Giải thích khi bị hỏi sâu:** `s1` là vector bí mật chính được nhân với `A`; `s2` là noise vector được cộng vào để tạo `t = A*s1 + s2`. Các hệ số của `s1`, `s2` nằm trong biên `[-eta, eta]`.

**Dẫn chứng code/test:** `ml_dsa/keygen.py::KeyGen_internal`; `ml_dsa/sampling.py::ExpandS`; `test/test_sampling.py::test_expand_s_dimensions_and_range`.

### Q13. Vì sao KeyGen dùng NTT?

**Câu hỏi:** Tại sao không nhân đa thức trực tiếp?

**Trả lời ngắn:** NTT giúp nhân đa thức/vector hiệu quả hơn trong ring ML-DSA.

**Giải thích khi bị hỏi sâu:** KeyGen cần tính `A*s1`. `A_hat` đã ở miền NTT, nên code đưa `s1` sang NTT bằng `NTT(p)`, nhân ma trận-vector bằng `MatrixVectorNTT`, rồi đưa kết quả về miền thường bằng `INTT`.

**Dẫn chứng code/test:** `ml_dsa/keygen.py::KeyGen_internal`; `ml_dsa/ntt.py`; `test/test_ntt.py`.

### Q14. `t = A*s1 + s2` được dùng để làm gì?

**Câu hỏi:** Sau khi tính `t`, vì sao phải tách `t1`, `t0`?

**Trả lời ngắn:** `t` được tách bằng `Power2Round` thành phần cao `t1` cho public key và phần thấp `t0` cho secret key.

**Giải thích khi bị hỏi sâu:** Public key chỉ cần `t1` để verify có thể kiểm tra chữ ký mà không lộ toàn bộ `t`. `t0` giữ trong secret key để signer tạo hint chính xác khi ký.

**Dẫn chứng code/test:** `ml_dsa/keygen.py::KeyGen_internal`; `ml_dsa/reduction.py::VectorPower2Round`; `test/test_reduction.py::test_power2round_decomposition_identity`.

### Q15. Public key chứa những gì?

**Câu hỏi:** Public key `pk` được encode từ thành phần nào?

**Trả lời ngắn:** Public key chứa `rho` và `t1` đã bit-pack.

**Giải thích khi bị hỏi sâu:** `rho` cho phép verifier tái tạo `A`; `t1` là phần public của `t`. Hàm `pkEncode(rho, t1, params)` đóng gói hai phần này thành bytes.

**Dẫn chứng code/test:** `ml_dsa/keygen.py::KeyGen_internal`; `ml_dsa/encoding.py::pkEncode`; `test/test_encoding.py::test_pk_encode_round_trip`.

### Q16. Secret key chứa những gì?

**Câu hỏi:** Secret key có chỉ chứa `s1` không?

**Trả lời ngắn:** Không. Secret key chứa `rho`, `K`, `tr`, `s1`, `s2`, và `t0`.

**Giải thích khi bị hỏi sâu:** `rho` tái tạo `A`; `K` dùng sinh randomness nội bộ khi ký; `tr = H(pk)` gắn khóa công khai vào message representative; `s1/s2/t0` là dữ liệu bí mật cần cho rejection sampling và hint.

**Dẫn chứng code/test:** `ml_dsa/keygen.py::KeyGen_internal`; `ml_dsa/encoding.py::skEncode`; `test/test_encoding.py::test_sk_encode_round_trip`.

### Q17. `tr = H(pk)` dùng để làm gì?

**Câu hỏi:** Vì sao secret key lưu thêm `tr`?

**Trả lời ngắn:** `tr` là hash của public key, dùng khi ký để ràng buộc message với đúng public key.

**Giải thích khi bị hỏi sâu:** Trong `Sign_internal`, code tính `mu = HShake256(tr + message_prime, 64)`. Verifier tự tính lại `tr = H(pk)` từ public key. Nếu dùng public key khác, `mu` khác và challenge tái tạo không khớp.

**Dẫn chứng code/test:** `ml_dsa/keygen.py::KeyGen_internal`; `ml_dsa/sign.py::Sign_internal`; `ml_dsa/verify.py::Verify_internal`; `test/test_pipeline.py::test_verify_fails_on_wrong_public_key`.

## C. Sign Pipeline

### Q18. `Sign` khác `Sign_internal` ở đâu?

**Câu hỏi:** Public `Sign` làm thêm việc gì trước khi vào thuật toán chính?

**Trả lời ngắn:** `Sign` format message với context/pre-hash và chọn `rnd`, sau đó gọi `Sign_internal`.

**Giải thích khi bị hỏi sâu:** `Sign_internal` nhận `message_prime` và `rnd` đã chuẩn bị. Public `Sign` hỗ trợ mode pure/HashML-DSA qua `FormatMessagePrime`, hỗ trợ deterministic mode bằng `rnd = 0^32`, hoặc random mode bằng `os.urandom(32)`.

**Dẫn chứng code/test:** `ml_dsa/sign.py::Sign`; `test/test_pipeline.py::test_deterministic_signing_is_repeatable`; `test_randomized_signing_produces_distinct_sigs`.

### Q19. `FormatMessagePrime` xử lý message như thế nào?

**Câu hỏi:** `message_prime` là gì?

**Trả lời ngắn:** `message_prime` là message đã được domain-separate bằng mode, độ dài context, context, và có thể cả OID + hash của message.

**Giải thích khi bị hỏi sâu:** Pure mode tạo `0x00 || len(ctx) || ctx || message`. HashML-DSA tạo `0x01 || len(ctx) || ctx || OID(PH) || PH(message)`. Điều này giúp cùng message nhưng khác context hoặc pre-hash mode tạo chữ ký khác nhau.

**Dẫn chứng code/test:** `ml_dsa/prehash.py::FormatMessagePrime`; `test/test_pipeline.py::test_context_string_separates_signatures`; `test_prehash_sign_verify_round_trip`.

### Q20. `ctx` có tác dụng gì?

**Câu hỏi:** Context có phải là một phần của message không?

**Trả lời ngắn:** Context là domain separation cho ứng dụng/giao thức, được đưa vào `message_prime` nhưng không phải nội dung message chính.

**Giải thích khi bị hỏi sâu:** Nếu hai ứng dụng dùng cùng key và cùng message nhưng context khác, chữ ký không dùng lẫn được. Code giới hạn context tối đa 255 byte theo format một byte độ dài.

**Dẫn chứng code/test:** `ml_dsa/prehash.py::FormatMessagePrime`; `ml_dsa/verify.py::Verify`; `test/test_pipeline.py::test_context_string_separates_signatures`; `test_verify_rejects_oversized_ctx`.

### Q21. Pure ML-DSA và HashML-DSA khác nhau thế nào?

**Câu hỏi:** Khi nào `pre_hash` là `None`, khi nào có giá trị?

**Trả lời ngắn:** `pre_hash=None` là pure mode, ký trực tiếp message trong `message_prime`; nếu truyền `sha256` hoặc hash được hỗ trợ thì dùng HashML-DSA.

**Giải thích khi bị hỏi sâu:** HashML-DSA hữu ích khi message lớn hoặc muốn ký digest theo chuẩn. Implementation hỗ trợ alias như `SHA2-256`, `SHA-256` thông qua normalize tên pre-hash.

**Dẫn chứng code/test:** `ml_dsa/prehash.py`; `test/test_pipeline.py::test_prehash_aliases_match_for_sign_and_verify`.

### Q22. Deterministic signing trong code hoạt động thế nào?

**Câu hỏi:** Vì sao cùng key và message có thể tạo cùng chữ ký?

**Trả lời ngắn:** Khi `deterministic=True`, `Sign` dùng `rnd = b"\x00" * 32`, nên quá trình sinh mask `y` lặp lại.

**Giải thích khi bị hỏi sâu:** `rnd` đi vào `rho_pp = H(K || rnd || mu)`. Nếu `K`, `rnd`, và `mu` giống nhau thì `rho_pp` giống nhau, nên chữ ký deterministic. Nếu truyền `rnd=os.urandom(32)`, chữ ký thường khác nhau.

**Dẫn chứng code/test:** `ml_dsa/sign.py::Sign`, `Sign_internal`; `test/test_pipeline.py::test_deterministic_signing_is_repeatable`; `test_randomized_signing_produces_distinct_sigs`.

### Q23. `rho_pp` dùng để làm gì?

**Câu hỏi:** Tại sao signer cần `rho_pp`?

**Trả lời ngắn:** `rho_pp` là seed để sinh vector mask `y` trong mỗi lần ký.

**Giải thích khi bị hỏi sâu:** `rho_pp = H(K + rnd + mu, 64)`, nên phụ thuộc secret `K`, randomness `rnd`, và message representative `mu`. Từ `rho_pp`, `ExpandMask` sinh `y`; nếu vòng ký bị reject, counter `kappa` tăng để lấy mask khác.

**Dẫn chứng code/test:** `ml_dsa/sign.py::Sign_internal`; `ml_dsa/sampling.py::ExpandMask`.

### Q24. `y`, `w`, `w1` là gì trong Sign?

**Câu hỏi:** Các biến này nằm ở bước nào của thuật toán?

**Trả lời ngắn:** `y` là mask bí mật tạm thời, `w = A*y`, và `w1` là phần high bits của `w` dùng để tạo challenge.

**Giải thích khi bị hỏi sâu:** Signer không đưa `w` thẳng vào signature. Thay vào đó, signer hash `mu || w1Encode(w1)` để sinh `c_tilde`. `w1` là phần verifier sẽ tái tạo xấp xỉ nhờ hint.

**Dẫn chứng code/test:** `ml_dsa/sign.py::Sign_internal`; `ml_dsa/reduction.py::VectorHighBits`; `ml_dsa/encoding.py::w1Encode`.

### Q25. Challenge `c_tilde` được tạo như thế nào?

**Câu hỏi:** Vì sao challenge phụ thuộc `mu` và `w1`?

**Trả lời ngắn:** `c_tilde = H(mu || w1Encode(w1))`, nên challenge bị ràng buộc với message, public key và commitment `w1`.

**Giải thích khi bị hỏi sâu:** Đây là phần Fiat-Shamir: từ transcript ký, hash sinh challenge. Nếu message hoặc public key đổi, `mu` đổi; nếu signature làm verifier tái tạo `w1` khác, challenge tái tạo cũng khác.

**Dẫn chứng code/test:** `ml_dsa/sign.py::Sign_internal`; `ml_dsa/verify.py::Verify_internal`; `test/test_pipeline.py::test_verify_fails_on_modified_message`.

### Q26. `SampleInBall` làm gì?

**Câu hỏi:** `c` khác gì `c_tilde`?

**Trả lời ngắn:** `c_tilde` là bytes challenge; `SampleInBall(c_tilde, params)` biến nó thành đa thức challenge thưa có đúng `tau` hệ số khác 0.

**Giải thích khi bị hỏi sâu:** Đa thức `c` dùng trong phép nhân với `s1`, `s2`, `t0`, `t1`. Số hệ số khác 0 được kiểm soát bởi tham số `tau`, giúp giữ chi phí và phân phối theo chuẩn.

**Dẫn chứng code/test:** `ml_dsa/sampling.py::SampleInBall`; `test/test_sampling.py::test_sample_in_ball_has_tau_nonzero_entries`.

### Q27. Vì sao Sign có rejection sampling?

**Câu hỏi:** Tại sao ký không trả signature ngay sau khi tính `z`?

**Trả lời ngắn:** Vì cần kiểm tra norm của `z`, `r0`, `ct0` và số hint để tránh lộ thông tin bí mật hoặc tạo signature không hợp lệ.

**Giải thích khi bị hỏi sâu:** Signer tính `z = y + c*s1`. Nếu `z` quá lớn, phân phối có thể phụ thuộc vào `s1`; nếu `r0` hoặc `ct0` quá lớn, verify/hint không còn đúng biên. Khi fail, code tăng `kappa` và sinh mask mới.

**Dẫn chứng code/test:** `ml_dsa/sign.py::Sign_internal`; `test/test_pipeline.py::test_sign_verify_round_trip`.

### Q28. `z` trong signature là gì?

**Câu hỏi:** `z` có phải secret key không?

**Trả lời ngắn:** Không. `z = y + c*s1` là response của chữ ký, được kiểm soát norm để không lộ `s1`.

**Giải thích khi bị hỏi sâu:** `z` phụ thuộc vào secret `s1`, nhưng nhờ mask `y` và rejection sampling, nó không trực tiếp tiết lộ `s1`. Trước khi encode, code chuyển `z` về dạng centered bằng `PolyToCentered`.

**Dẫn chứng code/test:** `ml_dsa/sign.py::Sign_internal`; `ml_dsa/poly.py::PolyToCentered`; `ml_dsa/encoding.py::sigEncode`.

### Q29. Hint `h` dùng để làm gì?

**Câu hỏi:** Vì sao signature cần thêm hint?

**Trả lời ngắn:** Hint giúp verifier phục hồi đúng `w1` mà không cần biết secret `t0`.

**Giải thích khi bị hỏi sâu:** Public key chỉ có `t1`, trong khi signer biết thêm `t0`. Khi verifier tính `A*z - c*t1*2^d`, kết quả có sai lệch nhỏ so với `w - c*s2 + c*t0`. Hint mã hóa nơi cần điều chỉnh high bits để verifier tái tạo cùng `w1`.

**Dẫn chứng code/test:** `ml_dsa/sign.py::VectorMakeHint`; `ml_dsa/verify.py::VectorUseHint`; `test/test_encoding.py::test_hint_bit_pack_round_trip`.

### Q30. `sigEncode` đóng gói những gì?

**Câu hỏi:** Signature bytes gồm các thành phần nào?

**Trả lời ngắn:** Signature gồm `c_tilde`, vector `z`, và hint `h`.

**Giải thích khi bị hỏi sâu:** `c_tilde` dùng để tái tạo challenge `c`; `z` là response; `h` giúp khôi phục `w1`. Các phần này được bit-pack để đạt đúng `sig_bytes` của từng parameter set.

**Dẫn chứng code/test:** `ml_dsa/encoding.py::sigEncode`; `test/test_encoding.py::test_sig_encode_round_trip`; `test/test_pipeline.py::test_sign_verify_round_trip`.

### Q31. Vì sao sửa một byte signature làm verify fail?

**Câu hỏi:** Signature có nhạy với thay đổi nhỏ không?

**Trả lời ngắn:** Có. Sửa signature làm `z`, `h` hoặc `c_tilde` khác, khiến challenge tái tạo không khớp hoặc decode/norm fail.

**Giải thích khi bị hỏi sâu:** Test sửa byte trong vùng sau `c_tilde`, tức thường ảnh hưởng `z`. Verify sẽ tính lại `w1_prime`, hash lại thành `c_tilde_prime`, rồi so với `c_tilde`; nếu khác thì trả `False`.

**Dẫn chứng code/test:** `ml_dsa/verify.py::Verify_internal`; `test/test_pipeline.py::test_verify_fails_on_modified_signature`.

## D. Verify Pipeline

### Q32. Verify bắt đầu bằng những kiểm tra nào?

**Câu hỏi:** Trước khi tính toán NTT, verifier kiểm tra gì?

**Trả lời ngắn:** Public `Verify` kiểm tra context, độ dài public key, độ dài signature, và format message; internal verify decode signature và kiểm tra popcount/norm.

**Giải thích khi bị hỏi sâu:** Các kiểm tra sớm giúp reject dữ liệu sai dạng mà không chạy toàn bộ pipeline. `sigDecode` có thể trả `None`; `Verify_internal` cũng reject nếu số hint vượt `omega` hoặc `||z||` vượt biên.

**Dẫn chứng code/test:** `ml_dsa/verify.py::Verify`, `Verify_internal`; `test/test_pipeline.py::test_verify_rejects_wrong_size_inputs`.

### Q33. Verify lấy `rho` và `t1` từ đâu?

**Câu hỏi:** Verifier có cần secret key để lấy ma trận `A` không?

**Trả lời ngắn:** Không. Verifier decode public key để lấy `rho` và `t1`, rồi dùng `rho` tái tạo `A`.

**Giải thích khi bị hỏi sâu:** Public key chứa đủ thông tin công khai: `rho` để sinh `A`, `t1` để tính biểu thức kiểm tra. Secret key chỉ cần cho signer.

**Dẫn chứng code/test:** `ml_dsa/verify.py::Verify_internal`; `ml_dsa/encoding.py::pkDecode`; `test/test_pipeline.py::test_sign_verify_round_trip`.

### Q34. Verifier tái tạo `mu` như thế nào?

**Câu hỏi:** Làm sao verifier biết `mu` signer đã dùng?

**Trả lời ngắn:** Verifier tự tính `tr = H(pk)` và `mu = H(tr || message_prime)`.

**Giải thích khi bị hỏi sâu:** `message_prime` được tạo cùng logic public `FormatMessagePrime`, còn `tr` lấy từ public key. Nếu message, context, pre-hash mode hoặc public key khác, `mu` khác.

**Dẫn chứng code/test:** `ml_dsa/verify.py::Verify_internal`; `ml_dsa/prehash.py::FormatMessagePrime`; `test/test_pipeline.py::test_verify_fails_on_wrong_public_key`.

### Q35. Verifier tái tạo challenge `c` thế nào?

**Câu hỏi:** `c` trong verify có được gửi trực tiếp không?

**Trả lời ngắn:** Không gửi trực tiếp. Signature chứa `c_tilde`, verifier dùng `SampleInBall(c_tilde, params)` để tái tạo `c`.

**Giải thích khi bị hỏi sâu:** Cách này giữ signature gọn: chỉ lưu challenge seed bytes thay vì toàn bộ đa thức. Từ seed này, cả signer và verifier sinh cùng sparse polynomial `c`.

**Dẫn chứng code/test:** `ml_dsa/verify.py::Verify_internal`; `ml_dsa/sampling.py::SampleInBall`.

### Q36. Biểu thức `A*z - c*t1*2^d` có ý nghĩa gì?

**Câu hỏi:** Vì sao verifier tính biểu thức này?

**Trả lời ngắn:** Nó là cách verifier tái tạo xấp xỉ commitment `w` của signer chỉ từ public key và signature.

**Giải thích khi bị hỏi sâu:** Signer có `z = y + c*s1` và public key có `t ≈ t1*2^d = A*s1 + s2 - t0`. Suy ra `A*z - c*t1*2^d` gần với `A*y - c*s2 + c*t0`, tức gần với dữ liệu signer dùng để tạo hint. Hint giúp chỉnh high bits về `w1`.

**Dẫn chứng code/test:** `ml_dsa/verify.py::Verify_internal`; `ml_dsa/sign.py::Sign_internal`.

### Q37. Vì sao verify dùng `UseHint`?

**Câu hỏi:** Nếu đã tính được `w_approx`, tại sao không hash trực tiếp nó?

**Trả lời ngắn:** Verifier cần high bits đúng như `w1` của signer; `UseHint` dùng hint để sửa high bits của `w_approx`.

**Giải thích khi bị hỏi sâu:** Do public key chỉ có `t1`, verifier thiếu `t0`. Sai lệch nhỏ có thể làm high bits lệch ở vài vị trí. Hint chỉ ra vị trí cần điều chỉnh mà không lộ secret.

**Dẫn chứng code/test:** `ml_dsa/verify.py::Verify_internal`; `ml_dsa/reduction.py::VectorUseHint`; `test/test_reduction.py::test_use_hint_inverts_perturbation`.

### Q38. Điều kiện cuối cùng của Verify là gì?

**Câu hỏi:** Khi nào verify trả `True`?

**Trả lời ngắn:** Khi `c_tilde` trong signature bằng `H(mu || w1Encode(w1_prime))` do verifier tự tái tạo.

**Giải thích khi bị hỏi sâu:** Đây là kiểm tra transcript Fiat-Shamir. Nếu message, public key, signature, context hoặc mode không khớp, giá trị hash cuối sẽ khác.

**Dẫn chứng code/test:** `ml_dsa/verify.py::Verify_internal`; `test/test_pipeline.py::test_sign_verify_round_trip`.

### Q39. Vì sao verify không cần secret key?

**Câu hỏi:** Nếu không có `s1`, `s2`, `t0`, verify dựa vào đâu?

**Trả lời ngắn:** Verify chỉ cần public key `rho,t1`, message, signature `c_tilde,z,h`, và các tham số scheme.

**Giải thích khi bị hỏi sâu:** Signature được thiết kế để chứng minh signer biết secret tương ứng mà không tiết lộ secret. `z` và `h` cho phép verifier dựng lại commitment cần kiểm tra; `t1` trong public key đại diện phần công khai của `A*s1+s2`.

**Dẫn chứng code/test:** `ml_dsa/verify.py::Verify_internal`; `test/test_pipeline.py::test_verify_fails_on_wrong_public_key`.

### Q40. Vì sao wrong public key làm verify fail?

**Câu hỏi:** Nếu signature đúng với message, đổi public key có sao không?

**Trả lời ngắn:** Fail, vì public key khác làm `tr`, `A`, `t1` khác, nên challenge tái tạo không khớp.

**Giải thích khi bị hỏi sâu:** Verify phụ thuộc cả `H(pk)` và biểu thức `A*z - c*t1*2^d`. Đổi `pk` thay đổi cả hai phần, nên signature không thể dùng với key khác.

**Dẫn chứng code/test:** `test/test_pipeline.py::test_verify_fails_on_wrong_public_key`.

### Q41. Verify xử lý pre-hash sai như thế nào?

**Câu hỏi:** Nếu ký bằng HashML-DSA nhưng verify bằng pure mode thì sao?

**Trả lời ngắn:** Verify trả `False` vì `message_prime` khác nên `mu` khác.

**Giải thích khi bị hỏi sâu:** Pure và HashML-DSA dùng domain separator khác (`0x00` vs `0x01`) và HashML-DSA còn thêm OID/hash. Vì vậy cùng message nhưng mode khác vẫn là transcript khác.

**Dẫn chứng code/test:** `test/test_pipeline.py::test_prehash_sign_verify_round_trip`; `ml_dsa/prehash.py::FormatMessagePrime`.

## E. Testing Và Câu Hỏi Phản Biện

### Q42. Test nào chứng minh pipeline end-to-end đúng?

**Câu hỏi:** Làm sao chứng minh KeyGen, Sign, Verify nối với nhau chạy đúng?

**Trả lời ngắn:** Test `test_sign_verify_round_trip` sinh key, ký message, kiểm tra signature length và verify trả `True` cho cả 3 parameter sets.

**Giải thích khi bị hỏi sâu:** Đây là test tích hợp cấp pipeline. Nó không chỉ test từng hàm lẻ mà kiểm tra `keygen -> sign -> verify` với API public `ML_DSA`.

**Dẫn chứng code/test:** `test/test_pipeline.py::test_sign_verify_round_trip`.

### Q43. Test nào chứng minh dữ liệu bị sửa sẽ bị phát hiện?

**Câu hỏi:** Nếu message hoặc signature bị sửa thì dự án kiểm tra ở đâu?

**Trả lời ngắn:** Có test sửa message, sửa signature, và dùng wrong public key; tất cả phải verify `False`.

**Giải thích khi bị hỏi sâu:** Ba test này bao phủ ba loại tấn công cơ bản: thay nội dung, thay chữ ký, thay khóa công khai. Chúng chứng minh check cuối `c_tilde == c_tilde_prime` nhạy với mọi phần transcript.

**Dẫn chứng code/test:** `test_verify_fails_on_modified_message`, `test_verify_fails_on_modified_signature`, `test_verify_fails_on_wrong_public_key`.

### Q44. Test nào chứng minh deterministic và randomized signing?

**Câu hỏi:** Dự án có kiểm tra chữ ký deterministic và random không?

**Trả lời ngắn:** Có. Deterministic mode ký lặp lại phải giống nhau; randomized signing với `os.urandom(32)` phải tạo chữ ký khác nhau.

**Giải thích khi bị hỏi sâu:** Hai test này kiểm tra đường đi của `rnd`: deterministic dùng toàn zero, randomized nhận entropy mới. Chúng cũng giúp phát hiện nếu code bỏ qua `rnd` hoặc dùng sai `rho_pp`.

**Dẫn chứng code/test:** `test/test_pipeline.py::test_deterministic_signing_is_repeatable`; `test_randomized_signing_produces_distinct_sigs`.

### Q45. Câu phản biện khó: implementation có bám FIPS 204 không?

**Câu hỏi:** Nếu giảng viên hỏi “em căn cứ đâu để nói code bám chuẩn?”, trả lời thế nào?

**Trả lời ngắn:** Code được chia theo các thuật toán FIPS: KeyGen, Sign, Verify, ExpandA/S, SampleInBall, Power2Round, encode/decode; test kiểm tra tham số, NTT, sampling, encoding, pipeline và regression hash.

**Giải thích khi bị hỏi sâu:** Không nên nói “đã chuẩn tuyệt đối” nếu chưa chạy KAT chính thức từ NIST. Nên nói implementation bám mapping trong `SPEC.md`, có test conformance nội bộ và regression để phát hiện thay đổi output. Nếu cần xác nhận cấp chuẩn, bước tiếp theo là đối chiếu official KAT.

**Dẫn chứng code/test:** `SPEC.md`; `test/test_params.py`; `test/test_ntt.py`; `test/test_sampling.py`; `test/test_encoding.py`; `test/test_pipeline.py`; `test/test_regression.py`.

## F. Câu Hỏi Bổ Sung Theo Nhóm

### Q46. Nếu dùng nhầm parameter set khi verify thì điều gì xảy ra?

**Câu hỏi:** Một signature tạo bởi `ML_DSA_44` có thể verify bằng `ML_DSA_65` không?

**Trả lời ngắn:** Không. Mỗi parameter set có kích thước key/signature và tham số nội bộ khác nhau, nên verify bằng scheme khác sẽ fail ngay từ kiểm tra size hoặc tái tạo sai transcript.

**Giải thích khi bị hỏi sâu:** Public API của mỗi object `ML_DSA` gắn với `params` cố định. `Verify` kiểm tra `len(pk) == params.pk_bytes` và `len(sig) == params.sig_bytes`. Nếu dữ liệu tình cờ qua size check thì các bước `ExpandA`, `SampleInBall`, `gamma2`, `omega`, `lambda` vẫn khác, nên challenge cuối không khớp.

**Dẫn chứng code/test:** `ml_dsa/api.py` tạo 3 object riêng; `ml_dsa/verify.py::Verify`; `test/test_params.py`; `test/test_pipeline.py::test_signing_with_all_three_schemes_using_same_seed_yields_distinct_keys`.

### Q47. Byte-size validation bảo vệ pipeline ở điểm nào?

**Câu hỏi:** Vì sao phải kiểm tra độ dài `pk`, `sk`, `sig` trước hoặc trong decode?

**Trả lời ngắn:** Vì encode/decode phụ thuộc layout byte chính xác; sai độ dài sẽ làm tách sai trường hoặc đọc thiếu dữ liệu.

**Giải thích khi bị hỏi sâu:** `pkDecode`, `skDecode`, `sigDecode` đều dựa vào offset tính từ tham số. Public `Verify` reject sớm nếu `pk/sig` sai size. Encoding tests cũng kiểm tra decode sai độ dài phải raise hoặc trả `None`.

**Dẫn chứng code/test:** `ml_dsa/verify.py::Verify`; `ml_dsa/encoding.py::pkDecode`, `skDecode`, `sigDecode`; `test/test_encoding.py::test_decode_functions_reject_wrong_lengths`.

### Q48. UI dùng API stateless có lợi gì?

**Câu hỏi:** Trong demo Streamlit, việc `ML_DSA` stateless giúp gì?

**Trả lời ngắn:** UI có thể giữ `pk`, `sk`, `sig`, `message` trong session riêng, còn object `ML_DSA_44/65/87` chỉ là service theo tham số.

**Giải thích khi bị hỏi sâu:** Nếu object scheme lưu secret key bên trong, UI dễ bị nhầm trạng thái giữa các lần keygen/sign hoặc giữa các parameter set. Với stateless API, UI truyền rõ secret key vào `sign` và public key vào `verify`, nên luồng dữ liệu minh bạch hơn.

**Dẫn chứng code/test:** `ml_dsa/api.py` chỉ lưu `params`; `app.py` dùng `st.session_state` để giữ dữ liệu UI; `test/test_ui_helpers.py::test_round_trip_using_helper_formatted_values`.

### Q49. Vì sao `sign` raise lỗi còn `verify` trả `False` là hợp lý?

**Câu hỏi:** Có bất nhất không khi `sign` và `verify` xử lý lỗi khác nhau?

**Trả lời ngắn:** Không. `sign` là thao tác cấu hình nội bộ của người giữ khóa; `verify` xử lý dữ liệu không tin cậy từ bên ngoài nên trả kết quả boolean.

**Giải thích khi bị hỏi sâu:** Nếu caller truyền `pre_hash="not-a-hash"` khi ký, đó là lỗi cấu hình cần sửa ngay nên raise `ValueError`. Khi verify gặp dữ liệu sai, caller chỉ cần biết chữ ký không hợp lệ, nên `False` là interface gọn và tránh lan exception ra ngoài.

**Dẫn chứng code/test:** `ml_dsa/sign.py::Sign`; `ml_dsa/verify.py::Verify`; `test/test_pipeline.py::test_invalid_prehash_name_rejected_without_changing_pure_api`.

### Q50. `pkEncode` và `skEncode` có vai trò gì trong KeyGen?

**Câu hỏi:** Sau khi tính xong các vector, vì sao KeyGen không trả trực tiếp vector?

**Trả lời ngắn:** Vì key chuẩn cần là bytes. `pkEncode` và `skEncode` biến các thành phần đa thức/seed thành định dạng byte ổn định.

**Giải thích khi bị hỏi sâu:** `pkEncode` đóng gói `rho,t1`; `skEncode` đóng gói `rho,K,tr,s1,s2,t0`. Các vector được bit-pack theo biên hệ số tương ứng, giúp key có đúng kích thước chuẩn và có thể round-trip bằng decode.

**Dẫn chứng code/test:** `ml_dsa/keygen.py::KeyGen_internal`; `ml_dsa/encoding.py::pkEncode`, `skEncode`; `test/test_encoding.py::test_pk_encode_round_trip`; `test_sk_encode_round_trip`.

### Q51. `t1` và `t0` khác nhau về vai trò bảo mật thế nào?

**Câu hỏi:** Tại sao `t1` public được còn `t0` phải nằm trong secret key?

**Trả lời ngắn:** `t1` là phần cao đủ để verify, còn `t0` là phần thấp cần cho signer tạo hint và không đưa vào public key.

**Giải thích khi bị hỏi sâu:** `Power2Round` tách `t` thành `t1*2^d + t0`. Public key dùng `t1` để giảm kích thước và giữ phần thấp ở phía secret. Trong Sign, `t0` được nhân với challenge để xác định hint; verifier không có `t0` nên cần `h`.

**Dẫn chứng code/test:** `ml_dsa/keygen.py::KeyGen_internal`; `ml_dsa/sign.py::Sign_internal`; `ml_dsa/reduction.py::Power2Round`; `test/test_reduction.py::test_power2round_decomposition_identity`.

### Q52. Vì sao `KeyGen_internal` bắt buộc seed đúng 32 byte?

**Câu hỏi:** Nếu seed dài hơn hoặc ngắn hơn thì có thể hash bình thường, tại sao vẫn reject?

**Trả lời ngắn:** Vì input chuẩn của KeyGen_internal là seed 32 byte; kiểm tra độ dài giúp test/KAT và API deterministic không mơ hồ.

**Giải thích khi bị hỏi sâu:** Nếu cho phép seed tùy ý, cùng một ý nghĩa “seed” có thể bị encode nhiều cách khác nhau. Bắt buộc 32 byte làm contract rõ ràng và tránh lỗi người dùng nhập seed sai.

**Dẫn chứng code/test:** `ml_dsa/keygen.py::KeyGen_internal`; `test/test_pipeline.py::test_keygen_deterministic_with_xi`; `test/test_ui_helpers.py::test_parse_hex_bytes_rejects_wrong_length`.

### Q53. Deterministic KeyGen có dùng cho production không?

**Câu hỏi:** Việc truyền `xi` cố định có nên dùng trong hệ thống thật không?

**Trả lời ngắn:** Chủ yếu dùng cho test, demo, KAT. Trong dùng thật nên để `KeyGen` tự lấy entropy bằng `os.urandom`.

**Giải thích khi bị hỏi sâu:** Deterministic keygen rất hữu ích để tái lập kết quả và regression test. Nhưng nếu seed bị lộ hoặc tái sử dụng không kiểm soát, key cũng bị tái tạo. Vì vậy production nên dùng CSPRNG của hệ điều hành.

**Dẫn chứng code/test:** `ml_dsa/keygen.py::KeyGen`; `test/test_pipeline.py::test_keygen_deterministic_with_xi`; `test/test_regression.py`.

### Q54. Verifier tái tạo ma trận public `A` từ đâu?

**Câu hỏi:** Nếu public key không lưu toàn bộ ma trận `A`, verifier lấy `A` bằng cách nào?

**Trả lời ngắn:** Public key lưu `rho`; verifier dùng `ExpandA(rho, params)` để tái tạo cùng ma trận `A_hat`.

**Giải thích khi bị hỏi sâu:** Đây là cách giảm kích thước public key. `rho` là seed công khai, nên bất kỳ ai cũng có thể sinh lại ma trận. KeyGen và Verify dùng cùng `ExpandA`, vì vậy nếu `rho` bị sửa thì `A` đổi và verify fail.

**Dẫn chứng code/test:** `ml_dsa/keygen.py::KeyGen_internal`; `ml_dsa/verify.py::Verify_internal`; `ml_dsa/sampling.py::ExpandA`; `test/test_sampling.py::test_expand_a_dimensions_and_determinism`.

### Q55. Vì sao `tr` nằm trong secret key dù verifier tự tính được?

**Câu hỏi:** Nếu `tr = H(pk)` thì tại sao secret key còn lưu `tr`?

**Trả lời ngắn:** Lưu `tr` giúp signer dùng trực tiếp hash public key đã tính khi keygen, còn verifier vẫn tự tính lại từ `pk`.

**Giải thích khi bị hỏi sâu:** `tr` không phải secret mới, nhưng nó nằm trong secret key encoding để signer không cần decode/tái hash public key riêng khi ký. Verify không tin vào secret key, nên vẫn tính `tr = H(pk)` từ public key được cung cấp.

**Dẫn chứng code/test:** `ml_dsa/keygen.py::KeyGen_internal`; `ml_dsa/sign.py::Sign_internal`; `ml_dsa/verify.py::Verify_internal`.

### Q56. `skDecode` trong Sign kiểm tra điều gì gián tiếp?

**Câu hỏi:** Khi ký, code lấy các thành phần secret key bằng cách nào?

**Trả lời ngắn:** `Sign_internal` gọi `skDecode(sk, params)` để lấy `rho`, `K`, `tr`, `s1`, `s2`, `t0` từ secret key bytes.

**Giải thích khi bị hỏi sâu:** Nếu secret key sai độ dài, `skDecode` raise `ValueError`. Sau decode, signer mới có đủ dữ liệu để tái tạo `A`, sinh `rho_pp`, nhân với `s1/s2/t0`, và tạo hint.

**Dẫn chứng code/test:** `ml_dsa/sign.py::Sign_internal`; `ml_dsa/encoding.py::skDecode`; `test/test_encoding.py::test_decode_functions_reject_wrong_lengths`.

### Q57. `w1Encode` có vai trò gì trong Fiat-Shamir?

**Câu hỏi:** Vì sao challenge hash không dùng trực tiếp list `w1`?

**Trả lời ngắn:** `w1Encode` chuyển `w1` thành bytes ổn định để hash transcript.

**Giải thích khi bị hỏi sâu:** Hash cần input byte canonical. Nếu hash list Python trực tiếp thì phụ thuộc biểu diễn runtime, không chuẩn. `w1Encode` bit-pack high bits theo tham số để signer và verifier có cùng bytes trước khi hash.

**Dẫn chứng code/test:** `ml_dsa/sign.py::Sign_internal`; `ml_dsa/verify.py::Verify_internal`; `ml_dsa/encoding.py::w1Encode`; `test/test_encoding.py::test_w1_encode_size`.

### Q58. `kappa` trong Sign dùng để làm gì?

**Câu hỏi:** Khi vòng ký bị reject, làm sao code không sinh lại đúng `y` cũ?

**Trả lời ngắn:** Code tăng `kappa` theo `params.l`; `ExpandMask(rho_pp, kappa, params)` sẽ sinh vector mask mới.

**Giải thích khi bị hỏi sâu:** `rho_pp` cố định cho một lần ký deterministic, nên cần counter để lấy mẫu khác khi reject. Tăng theo `l` đảm bảo các index polynomial trong vector mask không trùng với lần thử trước.

**Dẫn chứng code/test:** `ml_dsa/sign.py::Sign_internal`; `ml_dsa/sampling.py::ExpandMask`; `test/test_sampling.py::test_expand_mask_range_and_dimensions`.

### Q59. Các norm bound trong Sign bảo vệ điều gì?

**Câu hỏi:** Vì sao phải kiểm tra `VectorInfinityNorm(z)`, `r0`, và `ct0`?

**Trả lời ngắn:** Chúng đảm bảo signature nằm trong biên hợp lệ và phân phối không làm lộ secret.

**Giải thích khi bị hỏi sâu:** Nếu `z` quá lớn, response có thể tiết lộ thông tin về `s1`. Nếu `r0` hoặc `ct0` vượt biên, hint có thể không giúp verifier tái tạo `w1` đúng. Vì vậy signer reject và lấy mẫu lại.

**Dẫn chứng code/test:** `ml_dsa/sign.py::Sign_internal`; `ml_dsa/poly.py::VectorInfinityNorm`; `test/test_pipeline.py::test_sign_verify_round_trip`.

### Q60. Tham số `rnd` khác gì `xi`?

**Câu hỏi:** Cả KeyGen và Sign đều có dữ liệu ngẫu nhiên, chúng khác nhau thế nào?

**Trả lời ngắn:** `xi` là seed sinh khóa; `rnd` là randomness cho từng lần ký.

**Giải thích khi bị hỏi sâu:** `xi` quyết định `rho`, `rho_prime`, `K` và toàn bộ key pair. `rnd` đi vào `rho_pp = H(K || rnd || mu)` để tạo mask `y` cho signature. Cùng key nhưng khác `rnd` có thể tạo signature khác cho cùng message.

**Dẫn chứng code/test:** `ml_dsa/keygen.py::KeyGen`; `ml_dsa/sign.py::Sign`; `test/test_pipeline.py::test_randomized_signing_produces_distinct_sigs`.

### Q61. HashML-DSA có những hash nào được hỗ trợ?

**Câu hỏi:** Implementation hỗ trợ pre-hash tùy ý không?

**Trả lời ngắn:** Không tùy ý. Code hỗ trợ các tên trong `PH_OIDS`: `sha256`, `sha512`, `sha3_256`, `shake128`, `shake256`, kèm một số alias như `SHA-256`.

**Giải thích khi bị hỏi sâu:** HashML-DSA cần đưa OID của hash vào `message_prime`, nên hash phải nằm trong danh mục biết trước. Tên hash được normalize để `SHA2-256`, `SHA-256`, `sha256` cho cùng kết quả.

**Dẫn chứng code/test:** `ml_dsa/prehash.py::PH_OIDS`, `NormalizePHName`; `test/test_pipeline.py::test_prehash_aliases_match_for_sign_and_verify`.

### Q62. Nếu `pre_hash` sai thì Sign và Verify khác nhau thế nào?

**Câu hỏi:** Tại sao ký với pre-hash sai raise lỗi nhưng verify với pre-hash sai trả `False`?

**Trả lời ngắn:** Ký là cấu hình chủ động nên raise để sửa; verify là kiểm tra dữ liệu nên coi đó là không hợp lệ.

**Giải thích khi bị hỏi sâu:** `FormatMessagePrime` raise `ValueError` khi hash không nằm trong `PH_OIDS`. `Sign` không bắt lỗi này. `Verify` bắt `ValueError` và trả `False`, đúng với nguyên tắc verify không làm caller phải xử lý nhiều exception.

**Dẫn chứng code/test:** `ml_dsa/prehash.py::FormatMessagePrime`; `ml_dsa/sign.py::Sign`; `ml_dsa/verify.py::Verify`; `test/test_pipeline.py::test_invalid_prehash_name_rejected_without_changing_pure_api`.

### Q63. Vì sao signature không chứa message?

**Câu hỏi:** Khi verify cần message, tại sao không nhúng message vào signature luôn?

**Trả lời ngắn:** Signature chỉ chứng minh tính hợp lệ cho message được cung cấp; message là input riêng của verify.

**Giải thích khi bị hỏi sâu:** Signature chứa `c_tilde`, `z`, `h`. Message đi vào `message_prime`, rồi `mu`, rồi challenge hash. Nếu verifier dùng message khác, `mu` khác và check cuối fail. Không nhúng message giúp signature gọn và phù hợp cách ký thông thường.

**Dẫn chứng code/test:** `ml_dsa/encoding.py::sigEncode`; `ml_dsa/verify.py::Verify_internal`; `test/test_pipeline.py::test_verify_fails_on_modified_message`.

### Q64. Vì sao signer cần `t0_hat`?

**Câu hỏi:** Trong Sign, `t0` đã có trong secret key, tại sao lại đưa sang NTT?

**Trả lời ngắn:** Vì signer cần nhân challenge `c` với `t0` trong miền NTT cho hiệu quả và nhất quán với các phép nhân khác.

**Giải thích khi bị hỏi sâu:** `c_hat = NTT(c)`, còn `t0_hat = NTT(t0)`. Code dùng `_hat_pointwise_with_scalar_hat(c_hat, t0_hat)` rồi `INTT` để lấy `ct0`. Kết quả này dùng trong kiểm tra norm và tạo hint.

**Dẫn chứng code/test:** `ml_dsa/sign.py::Sign_internal`; `ml_dsa/ntt.py::MultiplyNTT`; `test/test_ntt.py::test_ntt_pointwise_matches_schoolbook`.

### Q65. `sigDecode` reject signature sai bằng cách nào?

**Câu hỏi:** Nếu signature bytes bị lỗi format, verifier phát hiện ở đâu?

**Trả lời ngắn:** `sigDecode` trả `None` nếu độ dài sai hoặc hint decode không hợp lệ; `Verify_internal` nhận `None` thì trả `False`.

**Giải thích khi bị hỏi sâu:** Signature có layout cố định gồm `c_tilde`, `z`, `h`. Nếu hint có index không tăng, tail không zero, hoặc độ dài không đúng, decode không tạo được cấu trúc hợp lệ. Verify reject trước khi tính toán sâu.

**Dẫn chứng code/test:** `ml_dsa/encoding.py::sigDecode`, `HintBitUnpack`; `ml_dsa/verify.py::Verify_internal`; `test/test_encoding.py::test_hint_bit_unpack_rejects_corrupt_data`; `test_decode_functions_reject_wrong_lengths`.

### Q66. `pkDecode` sai thì ảnh hưởng Verify thế nào?

**Câu hỏi:** Nếu public key thiếu byte hoặc bị cắt bớt thì verify ra sao?

**Trả lời ngắn:** Public `Verify` kiểm tra size và trả `False` trước khi gọi `pkDecode`.

**Giải thích khi bị hỏi sâu:** Điều này tránh decode offset sai. Nếu public key đúng size nhưng nội dung bị sửa, `pkDecode` vẫn ra `rho,t1`, nhưng `A`, `tr`, hoặc `t1` khác nên challenge cuối không khớp.

**Dẫn chứng code/test:** `ml_dsa/verify.py::Verify`; `ml_dsa/encoding.py::pkDecode`; `test/test_pipeline.py::test_verify_rejects_wrong_size_inputs`; `test_verify_fails_on_wrong_public_key`.

### Q67. `VectorInfinityNorm(z)` trong Verify kiểm tra điều gì?

**Câu hỏi:** Vì sao verifier kiểm tra norm của `z` dù signer đã kiểm tra?

**Trả lời ngắn:** Vì verifier không tin signature đầu vào; phải tự kiểm tra `z` vẫn nằm trong biên hợp lệ.

**Giải thích khi bị hỏi sâu:** Attacker có thể gửi signature tự tạo với `z` vượt biên. Nếu không reject, verifier có thể chấp nhận dữ liệu không đúng format chuẩn hoặc chạy các bước không cần thiết. Kiểm tra này là một điều kiện hợp lệ của signature.

**Dẫn chứng code/test:** `ml_dsa/verify.py::Verify_internal`; `ml_dsa/poly.py::VectorInfinityNorm`; `test/test_field_poly.py::test_vec_infinity_norm_picks_global_max`.

### Q68. `_scale_t1(t1, 1 << D)` dùng để làm gì?

**Câu hỏi:** Vì sao verify phải nhân `t1` với `2^d`?

**Trả lời ngắn:** Vì public key lưu phần high `t1`; để xấp xỉ lại `t`, verifier cần đưa `t1` về thang đo `t1*2^d`.

**Giải thích khi bị hỏi sâu:** KeyGen tách `t = t1*2^d + t0`. Verify không có `t0`, nhưng dùng `t1*2^d` để tính `A*z - c*t1*2^d`, rồi dùng hint để bù sai lệch high bits.

**Dẫn chứng code/test:** `ml_dsa/verify.py::_scale_t1`, `Verify_internal`; `ml_dsa/params.py::D`; `test/test_reduction.py::test_power2round_decomposition_identity`.

### Q69. Verify reject early có lợi gì?

**Câu hỏi:** Tại sao không để verify chạy hết rồi mới kết luận?

**Trả lời ngắn:** Reject early giúp tránh decode/tính toán nặng với dữ liệu chắc chắn sai và giữ API rõ ràng.

**Giải thích khi bị hỏi sâu:** Các bước NTT, matrix-vector multiplication tốn chi phí hơn kiểm tra độ dài, context, decode, norm. Vì vậy code reject sớm khi size sai, context quá dài, signature decode lỗi, hint vượt `omega`, hoặc `z` vượt biên.

**Dẫn chứng code/test:** `ml_dsa/verify.py::Verify`, `Verify_internal`; `test/test_pipeline.py::test_verify_rejects_oversized_ctx`; `test_verify_rejects_wrong_size_inputs`.

### Q70. Challenge mismatch thể hiện lỗi ở đâu?

**Câu hỏi:** Nếu mọi decode đều thành công nhưng signature vẫn sai, verify fail ở bước nào?

**Trả lời ngắn:** Fail ở so sánh cuối: `c_tilde == c_tilde_prime`.

**Giải thích khi bị hỏi sâu:** Verifier tái tạo `w1_prime` từ `A*z - c*t1*2^d` và hint, rồi hash `mu || w1Encode(w1_prime)`. Nếu transcript không đúng, `c_tilde_prime` khác `c_tilde`, nên trả `False`.

**Dẫn chứng code/test:** `ml_dsa/verify.py::Verify_internal`; `test/test_pipeline.py::test_verify_fails_on_modified_message`; `test_verify_fails_on_modified_signature`.

### Q71. Test NTT chứng minh phần nào của pipeline?

**Câu hỏi:** NTT tests liên quan gì đến KeyGen/Sign/Verify?

**Trả lời ngắn:** Chúng chứng minh phép biến đổi NTT/INTT và nhân pointwise dùng trong KeyGen, Sign, Verify hoạt động đúng.

**Giải thích khi bị hỏi sâu:** KeyGen dùng NTT để tính `A*s1`; Sign dùng NTT cho `A*y`, `c*s1`, `c*s2`, `c*t0`; Verify dùng NTT cho `A*z` và `c*t1`. Nếu NTT sai, pipeline có thể fail dù encode/sampling đúng.

**Dẫn chứng code/test:** `ml_dsa/ntt.py`; `test/test_ntt.py::test_ntt_round_trip`; `test_ntt_pointwise_matches_schoolbook`.

### Q72. Sampling tests chứng minh phần nào?

**Câu hỏi:** Tại sao cần test `ExpandA`, `ExpandS`, `ExpandMask`, `SampleInBall` riêng?

**Trả lời ngắn:** Vì sampling quyết định ma trận công khai, secret vectors, mask ký, và challenge; sai sampling làm toàn bộ pipeline sai.

**Giải thích khi bị hỏi sâu:** `ExpandA` phải deterministic và đúng kích thước; `ExpandS` phải sinh hệ số trong biên `eta`; `ExpandMask` phải sinh `y` đúng biên `gamma1`; `SampleInBall` phải có đúng `tau` hệ số khác 0. Các test riêng giúp khoanh vùng lỗi nếu pipeline fail.

**Dẫn chứng code/test:** `ml_dsa/sampling.py`; `test/test_sampling.py::test_expand_a_dimensions_and_determinism`; `test_expand_s_dimensions_and_range`; `test_expand_mask_range_and_dimensions`; `test_sample_in_ball_has_tau_nonzero_entries`.

### Q73. Encoding tests chứng minh điều gì?

**Câu hỏi:** Vì sao tài liệu nhấn mạnh encode/decode round-trip?

**Trả lời ngắn:** Vì API public dùng bytes; encode/decode sai sẽ làm key hoặc signature không thể verify dù toán học bên trong đúng.

**Giải thích khi bị hỏi sâu:** `pkEncode/pkDecode`, `skEncode/skDecode`, `sigEncode/sigDecode`, `HintBitPack/Unpack` là cầu nối giữa cấu trúc đa thức và bytes. Round-trip test đảm bảo dữ liệu không bị mất hoặc đổi dấu/bit khi đóng gói.

**Dẫn chứng code/test:** `ml_dsa/encoding.py`; `test/test_encoding.py::test_pk_encode_round_trip`; `test_sk_encode_round_trip`; `test_sig_encode_round_trip`; `test_hint_bit_pack_round_trip`.

### Q74. Regression hash giúp phát hiện lỗi gì?

**Câu hỏi:** Vì sao đã có round-trip tests còn cần regression hash?

**Trả lời ngắn:** Regression hash phát hiện output deterministic bị thay đổi ngoài ý muốn sau khi refactor.

**Giải thích khi bị hỏi sâu:** Một số thay đổi có thể vẫn làm sign/verify pass nhưng đổi bytes cụ thể của `pk`, `sk`, `sig`. Regression test đóng băng hash cho seed/message cố định để báo khi output thay đổi. Nếu thay đổi là do sửa bug có chủ đích thì mới cập nhật fixture.

**Dẫn chứng code/test:** `test/test_regression.py::test_deterministic_outputs_match_frozen_hashes`; `test/test_pipeline.py::test_deterministic_signing_is_repeatable`.

### Q75. Giới hạn lớn nhất của bộ test hiện tại là gì?

**Câu hỏi:** Nếu giảng viên hỏi “test như vậy đã đủ chứng minh chuẩn NIST chưa?”, trả lời sao?

**Trả lời ngắn:** Chưa nên khẳng định certified. Test hiện tại chứng minh implementation bám mapping và tự nhất quán; bước mạnh hơn là chạy official NIST KAT.

**Giải thích khi bị hỏi sâu:** Unit/integration/regression tests kiểm tra tham số, NTT, sampling, encoding và pipeline. Nhưng certified conformance cần so sánh với test vector chính thức. Vì vậy câu trả lời đúng là dự án đã có nền test tốt, còn official KAT là hướng xác nhận tiếp theo.

**Dẫn chứng code/test:** `SPEC.md`; `NIST.FIPS.204.pdf`; `test/test_params.py`; `test/test_pipeline.py`; `test/test_regression.py`.
