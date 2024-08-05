import React, { useState, useEffect, useRef } from "react";
import {
  StyleSheet,
  Text,
  TouchableOpacity,
  View,
  Image,
  ActivityIndicator,
  Dimensions,
} from "react-native";
import { CameraView, useCameraPermissions } from "expo-camera";
import * as tf from "@tensorflow/tfjs";
import { decodeJpeg } from "@tensorflow/tfjs-react-native";
import * as mobilenet from "@tensorflow-models/mobilenet";
import * as FileSystem from "expo-file-system";
import * as ImageManipulator from "expo-image-manipulator";

const SQUARE_SIZE = 640;
const SCREEN_WIDTH = Dimensions.get("window").width;
const SCREEN_HEIGHT = Dimensions.get("window").height;
const CAMERA_SIZE = Math.min(SCREEN_WIDTH, SCREEN_HEIGHT) - 40;

export default function App() {
  const [facing, setFacing] = useState("back");
  const [permission, requestPermission] = useCameraPermissions();
  const [model, setModel] = useState(null);
  const [predictions, setPredictions] = useState([]);
  const [isModelReady, setIsModelReady] = useState(false);
  const [capturedImage, setCapturedImage] = useState(null);
  const [isClassifying, setIsClassifying] = useState(false);
  const cameraRef = useRef(null);

  useEffect(() => {
    (async () => {
      await tf.ready();
      const loadedModel = await mobilenet.load();
      setModel(loadedModel);
      setIsModelReady(true);
      console.log("TensorFlow.js and MobileNet model loaded");
    })();
  }, []);

  if (!permission) {
    return <View />;
  }

  if (!permission.granted) {
    return (
      <View style={styles.container}>
        <Text style={styles.message}>
          We need your permission to show the camera
        </Text>
        <TouchableOpacity style={styles.button} onPress={requestPermission}>
          <Text style={styles.text}>Grant Permission</Text>
        </TouchableOpacity>
      </View>
    );
  }

  function toggleCameraFacing() {
    setFacing((current) => (current === "back" ? "front" : "back"));
  }

  const takePicture = async () => {
    if (cameraRef.current && isModelReady) {
      try {
        const photo = await cameraRef.current.takePictureAsync({
          quality: 1,
          base64: false,
          skipProcessing: true,
        });
        setCapturedImage(photo.uri);
        await classifyImage(photo.uri);
      } catch (error) {
        console.error("Error taking picture:", error);
      }
    } else {
      console.log("Camera or model not ready");
    }
  };

  const classifyImage = async (imageUri) => {
    try {
      setIsClassifying(true);
      setPredictions(["Classifying..."]);

      // Resize image
      const resizedImage = await ImageManipulator.manipulateAsync(
        imageUri,
        [{ resize: { width: 224, height: 224 } }],
        { format: "jpeg" }
      );

      const imgB64 = await FileSystem.readAsStringAsync(resizedImage.uri, {
        encoding: FileSystem.EncodingType.Base64,
      });
      const imgBuffer = tf.util.encodeString(imgB64, "base64").buffer;
      const raw = new Uint8Array(imgBuffer);
      const imageTensor = decodeJpeg(raw);

      const predictions = await model.classify(imageTensor);
      setPredictions(
        predictions
          .sort((a, b) => b.probability - a.probability)
          .slice(0, 5)
          .map(
            (pred) =>
              `${pred.className} (${(pred.probability * 100).toFixed(2)}%)`
          )
      );

      tf.dispose(imageTensor);
    } catch (error) {
      console.error("Classification error:", error);
      setPredictions(["Error classifying image"]);
    } finally {
      setIsClassifying(false);
    }
  };

  const handleGoBack = () => {
    setCapturedImage(null);
    setPredictions([]);
    setIsClassifying(false);
  };

  if (!isModelReady) {
    return (
      <View style={styles.loadingContainer}>
        <ActivityIndicator size="large" color="#0000ff" />
        <Text style={styles.loadingText}>Loading model...</Text>
      </View>
    );
  }

  return (
    <View style={styles.container}>
      {capturedImage ? (
        <View style={styles.previewContainer}>
          <Image source={{ uri: capturedImage }} style={styles.preview} />
          <Text style={styles.predictionText}>
            {isClassifying
              ? "Classifying..."
              : predictions.length > 0
              ? `Predictions:\n${predictions.join("\n")}`
              : "No predictions"}
          </Text>
          <TouchableOpacity
            style={styles.button}
            onPress={handleGoBack}
            disabled={isClassifying}
          >
            <Text style={styles.text}>
              {isClassifying ? "Please wait..." : "Take Another Picture"}
            </Text>
          </TouchableOpacity>
        </View>
      ) : (
        <View style={styles.cameraContainer}>
          <CameraView style={styles.camera} facing={facing} ref={cameraRef}>
            <View style={styles.overlay} />
          </CameraView>
          <View style={styles.buttonContainer}>
            <TouchableOpacity
              style={styles.button}
              onPress={toggleCameraFacing}
            >
              <Text style={styles.text}>Flip</Text>
            </TouchableOpacity>
            <TouchableOpacity style={styles.button} onPress={takePicture}>
              <Text style={styles.text}>Capture</Text>
            </TouchableOpacity>
          </View>
        </View>
      )}
    </View>
  );
}

const styles = StyleSheet.create({
  container: {
    flex: 1,
    justifyContent: "center",
    alignItems: "center",
    backgroundColor: "#000",
  },
  cameraContainer: {
    width: CAMERA_SIZE,
    height: CAMERA_SIZE + 100, // Extra space for buttons
    justifyContent: "flex-start",
    alignItems: "center",
  },
  camera: {
    width: CAMERA_SIZE,
    height: CAMERA_SIZE,
  },
  overlay: {
    flex: 1,
    borderWidth: 2,
    borderColor: "white",
    borderRadius: 10,
  },
  buttonContainer: {
    flexDirection: "row",
    justifyContent: "space-around",
    width: "100%",
    paddingTop: 20,
  },
  button: {
    backgroundColor: "rgba(255, 255, 255, 0.3)",
    padding: 15,
    borderRadius: 5,
    minWidth: 100,
    alignItems: "center",
  },
  text: {
    fontSize: 16,
    fontWeight: "bold",
    color: "white",
  },
  previewContainer: {
    flex: 1,
    justifyContent: "center",
    alignItems: "center",
    backgroundColor: "#000",
  },
  preview: {
    width: CAMERA_SIZE,
    height: CAMERA_SIZE,
    marginBottom: 20,
  },
  predictionText: {
    fontSize: 18,
    color: "white",
    marginBottom: 20,
    textAlign: "center",
  },
  loadingContainer: {
    flex: 1,
    justifyContent: "center",
    alignItems: "center",
    backgroundColor: "#000",
  },
  loadingText: {
    marginTop: 10,
    fontSize: 18,
    color: "white",
  },
});
