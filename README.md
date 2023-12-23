# MachineLearning_Final
52000835 - Nguyễn Thị Thanh Hiền

1. ## <a name="_toc154253899"></a>**Tìm hiểu, so sánh các phương pháp Optimizer**
   1. ### <a name="_toc154253900"></a>***Tổng quan về Optimizer***
- Định nghĩa

  Optimizer hay còn được gọi là thuật toán tối ưu, nó là một phần quan trọng trong việc huấn luyện mô hình học máy. Optimizer sử dụng để tối ưu hóa các hàm mất mát (loss) bằng cách xây dựng các mô hình Neural network với mục đích “học” được các features của dữ liệu đầu vào từ đó tìm cặp weights và bias phù hợp nhằm giảm thiếu giá trị của hàm mất mát. 

- Các thuật toán Optimizer phổ biến: Thuật toán thường được sử dụng trong phương pháp này là Gradient Descent, ngoài ra còn những biến thể khác của nó để có thể sử dụng để điều chỉnh trọng số. 
- Các biến thể của Gradient Descent là Stochastic Gradient Descent, Mini – batch Gradient Descent, Batch Gradient Descent. 

Cùng tìm hiểu những thuật toán phổ biến của Optimizer qua những nội dung dưới đây.
1. ### ` `***<a name="_toc154253901"></a>Các thuật toán phổ biến của Optimizer***
   1. ### <a name="_toc154253902"></a>***Gradient Descent***
- **Định nghĩa:** Thuật toán GD còn được gọi là giảm dần độ dốc, trong một bài toán tối ưu chúng ta thường tìm giá trị nhỏ nhất của một hàm số nào đó mà hàm phải đạt giá trị nhỏ nhất khi đạo hàm bằng 0. Nhưng không phải khi nào cũng có thể làm điều đó và thậm chí có nhiều trường lớp bất khả thi, vậy nên thuật toán GD sẽ giúp giải quyết vấn đề đó bằng cách chọn một nghiệm ngẫu nhiên cứ sau mỗi vòng lặp hay epoch thì cho nó tiến dần đến điểm cần tìm thêm một chút.
- **GD cho hàm 1 biến**

  Giả sử xt là điểm ta tìm được sau vòng lặp thứ t. Ta cần tìm một thuật toán để đưa xt về càng gần x càng tốt.

- Công thức cho tham số cập nhật: xt+1=xt-nf'(xt)

  Trong đó:

N là một số dương được gọi là learning rate 

Dấu (-) thể hiện việc ta phải đi ngược hướng với đạo hàm

- **GD cho hàm nhiều biến**

Giả sử ta cần tìm global minimum cho hàm f(θ) trong đó θ(*theta*) là một vector, thường được dùng để ký hiệu tập hợp các tham số của một mô hình cần tối ưu (trong Linear Regression thì các tham số chính là hệ số w). Đạo hàm của hàm số đó tại một điểm θ bất kỳ được ký hiệu là ∆θf(θ) (hình tam giác ngược đọc là *nabla*). Tương tự như hàm 1 biến, thuật toán GD cho hàm nhiều biến cũng bắt đầu bằng một điểm dự đoán θ0, sau đó, ở vòng lặp thứ t, quy tắc cập nhật là:

θt+1=θt-n∆θf(θt) 

Hoặc viết dưới dạng đơn giản hơn: θ=θ-n∆θf(θ)  

Quy tắc cần nhớ: **luôn luôn đi ngược hướng với đạo hàm**.

Công thức được áp dụng cho từng biến khi các biến khác cố định:

f'x≈ fx+ε-f(x-ε)2ε

- Tuy nhiên công thức này không được áp dụng vào tính đạo hàm vì độ phức tạp cao so với đạo hàm trực tiếp. Khi so sánh đạo hàm này với đạo hàm chính xác tính theo công thức, người ta thường giảm số chiều dữ liệu và giảm số điểm dữ liệu để thuận tiện cho tính toán. Một khi đạo hàm tính được rất gần với numerical gradient, chúng ta có thể tự tin rằng đạo hàm tính được là chính xác.
- **Cách thực thi**

  B1: Khởi tạo biến nội tại

B2: Đánh giá model dựa vào biến nội tại và hàm mất mát (loss function)

B3: Cập nhật các biến nội tại theo hướng tối ưu hàm mất mát 

B4: Lặp lại B2 và B3 cho tới khi thỏa điều kiện dừng

Công thức cập nhật GD được viết là:

θ= θ- n∆θ

Trong đó θ là tập các biến cần cập nhật, n là độ học (learning rate), ∆θfθ là gradient của hàm mất mát f theo tập θ.

- **Ưu điểm, nhược điểm**

  Ưu điểm: 

- Thuật toán gradient descent cơ bản, dễ hiểu. 
- Thuật toán đã giải quyết được vấn đề tối ưu model neural network bằng cách cập nhật trọng số sau mỗi vòng lặp.

Nhược điểm

- Vì đơn giản nên thuật toán Gradient Descent còn nhiều hạn chế như phụ thuộc vào nghiệm khởi tạo ban đầu và learning rate.
- Ví dụ 1 hàm số có 2 global minimum thì tùy thuộc vào 2 điểm khởi tạo ban đầu sẽ cho ra 2 nghiệm cuối cùng khác nhau.
- Tốc độ học quá lớn sẽ khiến cho thuật toán không hội tụ, quanh quẩn bên đích vì bước nhảy quá lớn; hoặc tốc độ học nhỏ ảnh hưởng đến tốc độ training

1. ### <a name="_toc154253903"></a>***Stochastic Gradient Descent (SGD)***
Stochastic Gradient Descent (SGD) là một thuật toán tối ưu hóa phổ biến được sử dụng trong quá trình huấn luyện mô hình học máy. Thuật toán này thuộc dạng tối ưu hóa gradient descent, nhưng có sự khác biệt quan trọng, đặc biệt là trong cách tính toán gradient và cập nhật trọng số.

Stochastic là 1 biến thể của Gradient Descent . Thay vì sau mỗi epoch chúng ta sẽ cập nhật trọng số (Weight) 1 lần thì trong mỗi epoch có N điểm dữ liệu chúng ta sẽ cập nhật trọng số N lần. Nhìn vào 1 mặt , SGD sẽ làm giảm đi tốc độ của 1 epoch. Tuy nhiên nhìn theo 1 hướng khác,SGD sẽ hội tụ rất nhanh chỉ sau vài epoch. Công thức SGD cũng tương tự như GD nhưng thực hiện trên từng điểm dữ liệu.

- **Nguyên lý hoạt động:**

  B1: Khởi tạo trọng số của mô hình

B2: Lặp qua dữ liệu: chọn ngẫu nhiên một điểm dữ liệu -> tính gradient của hàm mất mát tại điểm dữ liệu này -> Cập nhật trọng số theo công thức SGD

Công thức: θ= θ- α. ∆Jθ

B4: Lặp lại quá trình trên cho đến khi thỏa điều kiện dừng.

- **Ưu điểm và nhược điểm**

Ưu điểm: Thuật toán giải quyết được đối với cơ sở dữ liệu lớn mà GD không làm được. Thuật toán tối ưu này hiên nay vẫn hay được sử dụng.

Nhược điểm: Thuật toán vẫn chưa giải quyết được 2 nhược điểm lớn của gradient descent ( learning rate, điểm dữ liệu ban đầu ). Vì vậy ta phải kết hợp SGD với 1 số thuật toán khác như: Momentum, AdaGrad,..
1. ### <a name="_toc154253904"></a>***Momentum***
Momentum là một thuật toán được sinh ra để giải quyết các hạn chế của GD. 

Để hiểu về thuật toán thì chúng ta xét một ví dụ về quá trình lăn của viên bi như sau:

![](Aspose.Words.880b9015-7ec2-4f36-899d-6118757d479f.001.png)

Như hình b phía trên, nếu ta thả 2 viên bi tại 2 điểm khác nhau A và B thì viên bị A sẽ trượt xuống điểm C còn viên bi B sẽ trượt xuống điểm D, nhưng ta lại không mong muốn viên bi B sẽ dừng ở điểm D (local minimum) mà sẽ tiếp tục lăn tới điểm C (global minimum). Để thực hiện được điều đó ta phải cấp cho viên bi B 1 vận tốc ban đầu đủ lớn để nó có thể vượt qua điểm E tới điểm C.

Để thực hiện được ý tưởng này mà thuật toán Momentum ra đời

Công thức: vt= γrt-1+n∆θJ(θ)

Trong đó	

vt: vận tốc của viên bi

θt+1=θt-vt : là vị trí mới của viên bi

γ thường được chọn là một giá trị khoảng 0.9, vtlà  vận tốc tại thời điểm trước đó, ∆θJθchính là độ dốc của điểm trước đó. Sau đó vị trí mới của hòn bi được xác định như sau:

θ= θ- vt

![https://images.viblo.asia/full/77d900e2-0305-47cd-92e6-48604df4170c.png](Aspose.Words.880b9015-7ec2-4f36-899d-6118757d479f.002.png)

- **Ưu điểm, nhược điểm**
- Ưu điểm: Thuật toán tối ưu giải quyết được vấn đề: Gradient Descent không tiến được tới điểm global minimum mà chỉ dừng lại ở local minimum.
- Nhược điểm: Tuy momentum giúp hòn bi vượt dốc tiến tới điểm đích, tuy nhiên khi tới gần đích, nó vẫn mất khá nhiều thời gian giao động qua lại trước khi dừng hẳn, điều này được giải thích vì viên bi có đà.

  1. ### <a name="_toc154253905"></a>***Adagrad***
- Không như những thuật toán trên, learning rate hầu như giống nhau cho quá trình learning, adagrad coi learning rate cũng là một tham số
- Nó update tạo các update lớn với các dữ liệu khác biệt nhiều và các update nhỏ cho các dữ liệu ít khác biệt
- Adagrad chia learning rate với tổng bình phương của lịch sử biến thiên (đạo hàm)
- **Công thức**: ![](Aspose.Words.880b9015-7ec2-4f36-899d-6118757d479f.003.png)

  Trong đó:

ϵ là hệ số để tránh lỗi chia cho 0, default là 1e−8

G là một diagonal matrix nơi mà mỗi phần tử (i,i) là bình phương của đạo hàm vector tham số tại thời điểm t

- **Nguyên lý hoạt động**

  B1: Khởi tạo số lượng Gradient đã qua: 

- Cho mỗi tham số *θi*​, khởi tạo một biến *Gi*​ để theo dõi tổng bình phương của gradient của *θi*​ qua tất cả các lượt lặp trước đó.
- *Gi*​ được khởi tạo bằng 0.

B2: Tính gradient và cập nhật *Gi*​

Công thức cập nhật: θi = θi- αGi+ε. ∆J(θi)

Trong đó, *α* là learning rate, *ϵ* là một giá trị nhỏ (thường là 1*e*−8) để tránh chia cho 0.

- **Ưu điểm, nhược điểm**

  Ưu điểm: Một lơi ích dễ thấy của Adagrad là tránh việc điều chỉnh learning rate bằng tay, chỉ cần để tốc độ học default là 0.01 thì thuật toán sẽ tự động điều chỉnh.

Nhược điểm: Yếu điểm của Adagrad là tổng bình phương biến thiên sẽ lớn dần theo thời gian cho đến khi nó làm tốc độ học cực kì nhỏ, làm việc training trở nên đóng băng.

  1. ### <a name="_toc154253906"></a>***RMSprop***
- RMSprop, hay Root Mean Square Propagation, là một thuật toán tối ưu hóa được sử dụng trong quá trình huấn luyện mô hình học máy. Thuật toán này là một biến thể của Stochastic Gradient Descent (SGD), nhằm giảm bớt vấn đề của learning rate không đồng nhất trên các tham số khác nhau và giúp mô hình hội tụ nhanh hơn trong nhiều trường hợp.
- Thuật toán này sinh ra nhằm giải quyết vấn đề của Adagrad, nó giống với vector update đầu tiên của adadelta.
- Nguyên lý hoạt động

  B1: Điều chỉnh Learning rate cho từng tham số

B2: Damping Term để tránh việc chia cho độ lớn của gradient quá nhỏ gây ra nhiều và dao động không ổn định

B3: Cập nhật trọng số w của mô hình

*w*=*w*−​learning ratemean squared gradient​×gradient

Trong đó:

Mean squared gradient thường – 0.9 để cập nhật gradient.

mean squared gradient=0.9×mean squared gradient+0.1×gradient2

- **Ưu điểm, nhược điểm:**

  Ưu điểm: Ưu điểm rõ nhất của RMSprop là giải quyết được vấn đề tốc độ học giảm dần của Adagrad ( vấn đề tốc độ học giảm dần theo thời gian sẽ khiến việc training chậm dần, có thể dẫn tới bị đóng băng )

Nhược điểm: Thuật toán RMSprop có thể cho kết quả nghiệm chỉ là local minimum chứ không đạt được global minimum như Momentum. Vì vậy người ta sẽ kết hợp cả 2 thuật toán Momentum với RMSprop cho ra 1 thuật toán tối ưu Adam. Chúng ta sẽ trình bày nó trong phần sau.

1. ### <a name="_toc154253907"></a>***Adam***
- Giống với Adadelta và RMSprop, nó duy trì trung bình bình phương độ dốc (slope) quá khứ vt và cũng đồng thời duy trì trung bình độ dốc quá khứ mt, giống momentum.
- Trong khi momentum giống như một quả cầu lao xuống dốc, thì Adam lại giống như một quả cầu rất nặng và có ma sát (friction), nhờ vậy nó dễ dàng vượt qua local minimum và đạt tới điểm tối ưu nhất (flat minimum)
- Nó đạt được hiệu ứng Heavy Ball with Friction (HBF) nhờ vào hệ số (mt/ sqrt(vt))
- **Công thức:** 

![](Aspose.Words.880b9015-7ec2-4f36-899d-6118757d479f.004.png)

- **Ưu điểm, nhược điểm**

  Ưu điểm: 

- Tích hợp Momentum và RMSprop
- Hiệu quả cao trong tối ưu hóa, là lựa chọn tốt cho các bộ dữ liệu lớn.
- Tích hợp Regularization giúp tránh overfitting trong quá trình huấn luyện mô hình.

Nhược điểm: 

- Bộ nhớ phải đủ lớn để lưu trữ thông tin các lần cập nhật trước đó của thuật toán.
- Tùy chỉnh tham số quá tối ưu sẽ trở nên phức tạp và đòi hỏi sự hiểu biết sâu rộng về thuật toán. 
- Khả năng không ổn định trong vài trường hợp, đặc biệt khi không được cấu hình đúng cách cho bài toán cụ thể. 
  1. ### <a name="_toc154253908"></a>***So sánh***
So sánh module các thuật toán 

**SGD**

keras.optimizers.SGD(learning\_rate = 0.01, momentum = 0.0, nesterov = False)

**RMSprop**

keras.optimizers.RMSprop(learning\_rate = 0.001, rho = 0.9)

**Adagrad**

keras.optimizers.Adagrad(learning\_rate = 0.01)

**Adam**

keras.optimizers.Adam(

`   `learning\_rate = 0.001, beta\_1 = 0.9, beta\_2 = 0.999, amsgrad = False

)
## <a name="_toc154253909"></a>**2. Tìm hiểu về Continual Learning và Test Production khi xây dựng một giải pháp học máy để giải quyết một bài toán nào đó.**
### <a name="_toc154253910"></a>***2.1 Continual Learning***
- Continual Learning được gọi là quá trình học liên tục của một mô hình, đây được xem là bước cần thiết và quan trong của các mô hình học máy hướng đến trí tuệ nhân tạo. Quá trình này mô hình sẽ thực hiện học liên tục các dữ liệu mới trên nền tảng các dữ liệu cũ trước đó nhằm tránh sự lạc hậu của bộ dữ liệu và đưa ra dự đoán, kết quả chính xác, với mục tiêu làm cho bộ dữ liệu đó có thể được đưa vào môi trường sản xuất cách hoàn hảo nhất. 
- Tại sao cần học liên tục ?

  Khi dữ liệu thay đổi do bất kỳ một lý do nào đó như xu hướng hay các hành động của người dùng thì mô hình học của chúng ta buộc phải tiếp thu được dữ liệu mới đó để tránh sự lãng phí tài nguyên và thời gian để học lại từ đầu, chúng ta sẽ chỉ cần cho mô hình học thêm những phần dữ liệu được thay đổi mà không cần phải học lại từ đầu của mô hình.

- Thách thức phải đối mặt
- Dữ liệu thay đổi liên tục trong môi trường thực tế, mô hình nạp vào quá nhiều kiến thức mới khiến nó có thể quên đi kiến thức cũ trước đó.
- Sự chồng chéo giữa các lớp dữ liệu cũ và mới kiến việc phân loại khó khăn
- Duy trì mô hình cũng là một thách thức vì khi môi trường thay đổi thì các mẫu dữ liệu cũng thay đổi khiến các mô hình được đào tạo về dữ liệu dần trở nên lỗi thời. 
- Ưu điểm: 
- Ngăn sự lỗi thời của bộ dữ liệu
- Giúp bộ dữ liệu chính xác và hoạt động hiệu quả hơn
- Tiết kiệm thời gian tính toán và tiết kiệm tài nguyên lưu trữ
- Sự phản hồi liên tục giúp dữ liệu được cải thiện
- Nhược điểm:
- Khi bị quá tải có thể quên đi kiến thức cũ
- Vận hành khó khăn với dữ liệu đa nghiệm 
- Đòi hỏi vùng kiến thức lớn
### <a name="_toc154253911"></a>***2.2  Test Production***
Test Production còn gọi là sản xuất kiểm thử: là quá trình chuẩn bị và kiểm thử mô hình, dữ liệu phải là dữ liệu thực tế mà mô hình sẽ phải xử lý. 

Quy trình kiểm thử bao gồm:

- Chuẩn bị dữ liệu
- Kiểm thử hiệu suất
- Xử lý dữ liệu ngoại lai
- Kiểm thử an toàn và bảo mật
- Mục đích của việc này là nhằm đảm bảo rằng mô hình đã được huấn luyện và hoạt động cách tốt nhất và hiệu quả an toàn khi triển khai vào môi trường.
- Khu vực trọng điểm để thiết lập kiểm thử bao gồm:
- Hệ thống và ứng dụng
- Dữ liệu kiểm thử
- Máy chủ cơ sở dữ liệu
- Môi trường chạy 
- Hệ điều hàng
- Mạng 
- Thách thức: 
- Cần dữ liệu thực tế mà mô hình phải đối mặt, đảm bảo sự chính xác đáng tin cậy của bộ dữ liệu.
- Khả năng mở rộng và hiệu suất, đặc biệt là khi mô hình phải xử lý lượng lớn dữ liệu và có nhiều người truy cập cùng lúc.
- An toàn và bảo mật là thách thức lớn
- Mô hình cần phải tích hợp chặt chẽ với hệ thống và cố gắng để không xảy ra xung đột.
- Ưu điểm: Ngăn chặn sự tấn công và rủi ro bộ dữ liệu và ngăn chặn các kiểm họa và khả năng phục hồi tốt hơn.
- Nhược điểm: Tốn thời gian, đặc biệt với các bộ dữ liệu lớn. 
### <a name="_toc154253912"></a>***2.3 Ví dụ***
**Continual learning**

Ví dụ: tất cả chúng ta đều đã trải nghiệm hệ thống đề xuất rất thành công của Netflix cho “Up Next”. Hệ thống đề xuất của Netflix đề xuất một chương trình ngay sau khi tập cuối cùng của bạn kết thúc và thường khó cưỡng lại khi giây giảm xuống. Kiểu mẫu đó trong sản xuất là thứ cần được đào tạo lại định kỳ vì thị trường có phim mới, thị hiếu mới, xu hướng mới. Với việc học liên tục, mục tiêu là sử dụng dữ liệu sắp có và sử dụng dữ liệu đó để tự động đào tạo lại mô hình, nhờ đó bạn thực sự có thể đạt được độ chính xác cao và giữ lại các mô hình có hiệu suất cao.

**Test Production**

Bài toán: Dự đoán Tình Trạng Giao Thông

Với mục đích xây dựng một mô hình học máy để dự đoán tình trạng giao thông trên các tuyến đường trong một thành phố dựa trên dữ liệu từ các cảm biến và thông tin lưu lượng giao thông. 

- Chuẩn bị dữ liệu: độ dày lưu lượng xe, tốc độ, thời gian, các biến khác có thể ảnh hưởng đến tình trạng giao thông.
- Chuẩn bị mô hình: chọn mô hình Random Forest hoặc Gradient boosting. Tối ưu hóa tham số mô hình trên tập huấn luyện.
- Kiểm thử hiệu suất: Đo lường các chỉ số như độ chính xác, độ nhạy, độ đặc biệt để đảm bảo rằng mô hình đáp ứng đúng yêu cầu.
- Kiểm thử an toàn và bảo mật: Đảm bảo rằng mô hình không tạo ra dự đoán giao thông không an toàn hoặc không phù hợp với môi trường thực tế. Kiểm tra tính an toàn và bảo mật của mô hình.
- Kiểm thử tích hợp hệ thống: kiểm tra tính tương tác của nó với các thành phần khác của hệ thống.
- Xử lý các biến ngoại lai
- Kiểm thử quyền và quản lý truy cập: đảm bảo rằng chỉ những người cần thiết có thể truy cập và sử dụng mô hình.
- Kiểm thử lặp lại: đảm bảo mô hình đáng tin cậy và đưa ra dự đoán nhất quán trong nhiều lần triển khai
- Kiểm thử hiệu suất thực tế: Triển khai mô hình vào môi trường thực tế và kiểm tra hiệu suất thực tế của nó theo thời gian. Đánh giá các chỉ số hiệu suất và điều chỉnh mô hình khi cần thiết.

# **<a name="_toc154253913"></a>TÀI LIỆU THAM KHẢO**
#
<https://vncoder.vn/bai-hoc/tong-hop-model-phan-1-431>

<https://machinelearningcoban.com/2017/01/12/gradientdescent/>

<https://viblo.asia/p/optimizer-hieu-sau-ve-cac-thuat-toan-toi-uu-gdsgdadam-Qbq5QQ9E5D8>





