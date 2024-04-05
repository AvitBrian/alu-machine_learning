import 'package:flutter/material.dart';

class HomeScreen extends StatelessWidget {
  const HomeScreen({super.key});

  @override
  Widget build(BuildContext context) {
    return Scaffold(
      
      appBar: AppBar(
        elevation: 10,
          title: const Text("Machine Learning Gee"),
          centerTitle: true,
        ),
        body: const Center(
          child: Text('Our Model design goes here...'),
        ),
    );
  }
}