class Prediction {
  final dynamic data;

  Prediction({required this.data});
  factory Prediction.fromJson(Map<String, dynamic> json) =>
      Prediction(data: json['body']);
}
