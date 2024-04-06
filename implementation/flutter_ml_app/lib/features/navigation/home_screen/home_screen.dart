import 'dart:convert';
import 'dart:ui';

import 'package:flutter/material.dart';
import 'package:flutter/services.dart'; // Import services package
import 'package:http/http.dart' as http;
import 'package:intl/intl.dart'; // Import intl package

class HomeScreen extends StatefulWidget {
  const HomeScreen({Key? key});

  @override
  State<HomeScreen> createState() => _HomeScreenState();
}

class _HomeScreenState extends State<HomeScreen> {
  String? tvMarketingSales;
  String? prediction;
  late final TextEditingController _textController;

  @override
  void initState() {
    super.initState();
    _textController = TextEditingController();
  }

  @override
  void dispose() {
    _textController.dispose();
    super.dispose();
  }

  Future<double?> fetchPrediction(String? tvMarketingSales) async {
    try {
      final uri = Uri.https("avitsmodel.onrender.com", "/predict");
      final response = await http.post(
        uri,
        headers: {"Content-Type": "application/json"},
        body: json.encode({"predicted_sales": tvMarketingSales}),
      );
      double? predictedValue;
      if (response.statusCode == 200) {
        final Map<String, dynamic> data = json.decode(response.body);
        predictedValue = data["predicted_sales"];
      }
      return predictedValue;
    } catch (e) {
      throw Exception("Failed to fetch prediction: $e");
    }
  }

  Future<void> handlePredictButtonPress() async {
    setState(() {
      prediction = null;
    });

    final String sales = _textController.text
        .trim()
        .replaceAll(',', ''); // Remove commas before sending
    if (sales.isNotEmpty) {
      try {
        final double? fetchedPrediction = await fetchPrediction(sales);
        setState(() {
          prediction = fetchedPrediction != null
              ? NumberFormat.decimalPattern()
                  .format(fetchedPrediction) // Format with commas
              : null;
        });
      } catch (e) {
        print(e);
        ScaffoldMessenger.of(context).showSnackBar(SnackBar(
          content: Text("Failed to fetch prediction: $e"),
          backgroundColor: Colors.red,
        ));
      }
    } else {
      ScaffoldMessenger.of(context).showSnackBar(const SnackBar(
        content: Text("Please enter TV Marketing Sales value"),
        backgroundColor: Colors.red,
      ));
    }
  }

  @override
  Widget build(BuildContext context) {
    return Scaffold(
      appBar: AppBar(
        title: const Text(
          "Predictarium",
          style: TextStyle(color: Colors.black),
        ),
        centerTitle: true,
        backgroundColor: Colors.white,
        elevation: 0,
      ),
      backgroundColor: Colors.white,
      body: CustomScrollView(
        slivers: [
          SliverFillRemaining(
              hasScrollBody: false,
              child: Padding(
                padding: const EdgeInsets.all(20.0),
                child: SafeArea(
                  child: Column(
                    mainAxisAlignment: MainAxisAlignment.center,
                    crossAxisAlignment: CrossAxisAlignment.start,
                    children: [
                      Expanded(
                        flex: 1,
                        child: Stack(
                          children: [
                            Container(
                              decoration: BoxDecoration(
                                image: const DecorationImage(
                                  image: AssetImage("assets/faq.png"),
                                  fit: BoxFit.contain,
                                ),
                                borderRadius: BorderRadius.circular(20),
                              ),
                              padding: const EdgeInsets.all(20),
                            ),
                            if (prediction != null)
                              Container(
                                decoration: BoxDecoration(
                                  borderRadius: BorderRadius.circular(20),
                                ),
                                padding: const EdgeInsets.all(20),
                                child: BackdropFilter(
                                  filter:
                                      ImageFilter.blur(sigmaX: 10, sigmaY: 10),
                                  child: Container(
                                    color: Colors.transparent,
                                    child: Column(
                                      mainAxisAlignment:
                                          MainAxisAlignment.center,
                                      children: [
                                        Center(
                                          child: Text(
                                            "\$ $prediction",
                                            style: TextStyle(
                                              fontSize: 56,
                                              fontWeight: FontWeight.bold,
                                              foreground: Paint()
                                                ..style = PaintingStyle.stroke
                                                ..strokeWidth = .3
                                                ..color = Colors
                                                    .black, // Outline color
                                              shadows: const [
                                                Shadow(
                                                  blurRadius: 1,
                                                  offset: Offset(
                                                      .5, .5), // Shadow offset
                                                  color: Colors.blue,
                                                )
                                              ],
                                            ),
                                          ),
                                        ),
                                      ],
                                    ),
                                  ),
                                ),
                              ),
                          ],
                        ),
                      ),
                      const SizedBox(height: 20),
                      Expanded(
                        flex: 1,
                        child: Container(
                          
                          decoration: BoxDecoration(
                            color: Colors.blue.withOpacity(0.1),
                            borderRadius: BorderRadius.circular(20),
                          ),
                          padding: const EdgeInsets.all(20),
                          child: Column(
                            mainAxisAlignment: MainAxisAlignment.spaceEvenly,
                            children: [
                              const Text(
                                """\"TV marketing expenses greatly affect sales by boosting brand visibility,engaging consumers, and driving purchases. Strategic allocation of marketing funds to TV campaigns expands audience reach,enhances brand recognition, and ultimately drives sales growth.\"""",
                                style: TextStyle(fontSize: 12),
                                textAlign: TextAlign.justify,
                              ),
                              const SizedBox(
                                height: 10,
                              ),
                              SizedBox(
                                height: 60,
                                width:
                                    MediaQuery.of(context).size.width / 2.7,
                                child: TextFormField(
                                  controller: _textController,
                                  keyboardType: TextInputType.number,
                                  textAlign: TextAlign.center,
                                  inputFormatters: <TextInputFormatter>[
                                    FilteringTextInputFormatter.digitsOnly,
                                    ThousandsFormatter(), // Custom formatter
                                  ],
                                  decoration: InputDecoration(
                                    prefix: const Text("\$"),
                                    hintText: "0.00",
                                    filled: true,
                                    fillColor: Color.fromARGB(13, 186, 185, 185).withOpacity(0.2),
                                    border: OutlineInputBorder(
                                      borderRadius: BorderRadius.circular(15),
                                      borderSide: BorderSide.none,
                                    ),
                                  ),
                                  onChanged: (value) {
                                    setState(() {
                                      tvMarketingSales = value;
                                    });
                                  },
                                ),
                              ),
                              const SizedBox(height: 5),
                              ElevatedButton(
                                onPressed: handlePredictButtonPress,
                                style: ElevatedButton.styleFrom(
                                  backgroundColor: Colors.blue,
                                  shape: RoundedRectangleBorder(
                                    borderRadius: BorderRadius.circular(18),
                                  ),
                                  padding: const EdgeInsets.symmetric(
                                      vertical: 16, horizontal: 20),
                                ),
                                child: Text(
                                  "Predict Sales",
                                  style: TextStyle(
                                    fontSize: 16,
                                    fontWeight: FontWeight.bold,
                                    color: Colors.white.withOpacity(0.7),
                                  ),
                                ),
                              ),
                            ],
                          ),
                        ),
                      ),
                    ],
                  ),
                ),
              ))
        ],
      ),
    );
  }
}

class ThousandsFormatter extends TextInputFormatter {
  @override
  TextEditingValue formatEditUpdate(
      TextEditingValue oldValue, TextEditingValue newValue) {
    final regEx = RegExp(r'(\d{1,3})(?=(\d{3})+(?!\d))');
    String newText =
        newValue.text.replaceAllMapped(regEx, (Match match) => '${match[1]},');
    return newValue.copyWith(
        text: newText,
        selection: TextSelection.collapsed(offset: newText.length));
  }
}
