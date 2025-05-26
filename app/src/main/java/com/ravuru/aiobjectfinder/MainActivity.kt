package com.ravuru.aiobjectfinder

import android.Manifest
import android.content.Context
import android.content.pm.PackageManager
import android.graphics.Bitmap
import android.graphics.BitmapFactory
import android.os.Bundle
import android.util.Log
import android.widget.Toast
import androidx.activity.ComponentActivity
import androidx.activity.compose.setContent
import androidx.camera.core.*
import androidx.camera.lifecycle.ProcessCameraProvider
import androidx.camera.view.PreviewView
import androidx.compose.foundation.layout.*
import androidx.compose.material3.*
import androidx.compose.runtime.*
import androidx.compose.ui.Alignment
import androidx.compose.ui.Modifier
import androidx.compose.ui.platform.LocalContext
import androidx.compose.ui.platform.LocalLifecycleOwner
import androidx.compose.ui.unit.dp
import androidx.compose.ui.viewinterop.AndroidView
import androidx.core.app.ActivityCompat
import androidx.core.content.ContextCompat
import org.tensorflow.lite.support.image.TensorImage
import org.tensorflow.lite.support.label.Category
import org.tensorflow.lite.task.vision.detector.ObjectDetector
import java.nio.ByteBuffer
import java.util.concurrent.ExecutorService
import java.util.concurrent.Executors

class MainActivity : ComponentActivity() {

    private lateinit var imageCapture: ImageCapture
    private lateinit var cameraExecutor: ExecutorService

    override fun onCreate(savedInstanceState: Bundle?) {
        super.onCreate(savedInstanceState)

        if (allPermissionsGranted()) {
            setContent {
                CameraCaptureScreen()
            }
        } else {
            ActivityCompat.requestPermissions(
                this, REQUIRED_PERMISSIONS, REQUEST_CODE_PERMISSIONS
            )
        }

        cameraExecutor = Executors.newSingleThreadExecutor()
    }

    private fun allPermissionsGranted() = REQUIRED_PERMISSIONS.all {
        ContextCompat.checkSelfPermission(baseContext, it) == PackageManager.PERMISSION_GRANTED
    }

    override fun onRequestPermissionsResult(
        requestCode: Int, permissions: Array<out String>, grantResults: IntArray
    ) {
        super.onRequestPermissionsResult(requestCode, permissions, grantResults)
        if (requestCode == REQUEST_CODE_PERMISSIONS && allPermissionsGranted()) {
            setContent {
                CameraCaptureScreen()
            }
        } else {
            Toast.makeText(this, "Camera permission denied", Toast.LENGTH_SHORT).show()
            finish()
        }
    }

    companion object {
        private const val REQUEST_CODE_PERMISSIONS = 10
        private val REQUIRED_PERMISSIONS = arrayOf(Manifest.permission.CAMERA)
    }

    @Composable
    fun CameraCaptureScreen() {
        val context = LocalContext.current
        val lifecycleOwner = LocalLifecycleOwner.current
        val previewView = remember { PreviewView(context) }

        var isObjectDetected by remember { mutableStateOf(false) }
        var detectionResult by remember { mutableStateOf<DetectionResult?>(null) }
        var isProcessing by remember { mutableStateOf(false) }
        var hasCaptured by remember { mutableStateOf(false) }

        val captureText = if (isProcessing) "Processing..." else if (hasCaptured) "Capture Again" else "Capture"

        Column(
            modifier = Modifier.fillMaxSize(),
            verticalArrangement = Arrangement.SpaceBetween,
            horizontalAlignment = Alignment.CenterHorizontally
        ) {
            AndroidView(factory = { previewView }, modifier = Modifier
                .fillMaxWidth()
                .weight(1f))

            Button(
                onClick = {
                    isProcessing = true
                    takePhoto(context) {
                        val (isDetected, detectedObj) = it
                        isObjectDetected = isDetected
                        detectionResult = detectedObj
                        isProcessing = false
                        hasCaptured = true
                    }
                },
                enabled = !isProcessing,
                modifier = Modifier
                    .width(200.dp)
                    .padding(16.dp)
            ) {
                Text(text = captureText)
            }

            val resultText = when {
                !hasCaptured -> "📷 Ready to capture. Click the button below."
                detectionResult != null && isObjectDetected -> {
                    "✅ Target Object Found!\n" +
                            "Label: ${detectionResult?.label}\n" +
                            "Score: ${"%.2f".format(detectionResult?.score?.times(100) ?: 0f)}%\n" +
                            "Category: ${detectionResult?.category?.label}"
                }
                detectionResult != null -> {
                    "✅ Object Detected (Not a Target)\n" +
                            "Label: ${detectionResult?.label}\n" +
                            "Score: ${"%.2f".format(detectionResult?.score?.times(100) ?: 0f)}%\n" +
                            "Category: ${detectionResult?.category?.label}"
                }
                else -> {
                    "📷 No object detected. Try again."
                }
            }

            Text(
                text = resultText,
                color = if (!hasCaptured) MaterialTheme.colorScheme.primary else MaterialTheme.colorScheme.onSurface,
                modifier = Modifier
                    .fillMaxWidth()
                    .align(Alignment.CenterHorizontally)
                    .padding(16.dp)
            )
        }

        LaunchedEffect(Unit) {
            val cameraProvider = ProcessCameraProvider.getInstance(context).get()

            val preview = Preview.Builder().build().also {
                it.setSurfaceProvider(previewView.surfaceProvider)
            }

            imageCapture = ImageCapture.Builder().build()

            val cameraSelector = CameraSelector.DEFAULT_BACK_CAMERA

            try {
                cameraProvider.unbindAll()
                cameraProvider.bindToLifecycle(
                    lifecycleOwner, cameraSelector, preview, imageCapture
                )
            } catch (e: Exception) {
                Log.e("CameraX", "Use case binding failed", e)
            }
        }
    }


    private fun takePhoto(context: Context, invoke: (Pair<Boolean, DetectionResult?>) -> Unit) {
        val imageCapture = imageCapture ?: return

        imageCapture.takePicture(
            ContextCompat.getMainExecutor(context),
            object : ImageCapture.OnImageCapturedCallback() {
                override fun onCaptureSuccess(image: ImageProxy) {
                    val bitmap = imageProxyToBitmap(image)
                    val result = runObjectDetection(bitmap, context)
                    image.close()
                    invoke(result)
                }

                override fun onError(exception: ImageCaptureException) {
                    Toast.makeText(context, "Capture failed: ${exception.message}", Toast.LENGTH_SHORT).show()
                }
            })
    }

    private fun imageProxyToBitmap(image: ImageProxy): Bitmap {
        val buffer: ByteBuffer = image.planes[0].buffer
        val bytes = ByteArray(buffer.remaining())
        buffer.get(bytes)
        return BitmapFactory.decodeByteArray(bytes, 0, bytes.size)
    }

    private fun runObjectDetection(bitmap: Bitmap, context: Context): Pair<Boolean, DetectionResult?> {
        val options = ObjectDetector.ObjectDetectorOptions.builder()
            .setMaxResults(5)
            .setScoreThreshold(0.5f)
            .build()

        val detector = ObjectDetector.createFromFileAndOptions(
            context,
            "efficientdet_lite0.tflite",
            options
        )

        val image = TensorImage.fromBitmap(bitmap)
        val results = detector.detect(image)

        if (results.isEmpty()) {
            return Pair(false, null)
        }

        val result = results.firstOrNull()
        val category = result?.categories?.firstOrNull()
        val label = category?.label ?: "Unknown"
        val score = category?.score ?: 0f

        Log.i("ObjectDetection", "Label: $label, Score: $score Category: $category")
        val detectionResult = DetectionResult(label, score, category)

        val targetFound = label.contains("bin", true)
                || label.contains("trash", true)
                || label.contains("recycle", true)
                || label.contains("chair", true)
                || label.contains("toilet", true)

        return Pair(targetFound, detectionResult)
    }

    data class DetectionResult(val label: String, val score: Float, val category: Category?)
}
