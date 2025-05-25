package com.ravuru.aiobjectfinder

import android.Manifest
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
import androidx.compose.ui.Modifier
import androidx.compose.ui.platform.LocalContext
import androidx.compose.ui.platform.LocalLifecycleOwner
import androidx.compose.ui.unit.dp
import androidx.compose.ui.viewinterop.AndroidView
import androidx.core.app.ActivityCompat
import androidx.core.content.ContextCompat
import org.tensorflow.lite.support.image.TensorImage
import org.tensorflow.lite.task.vision.detector.ObjectDetector
import java.nio.ByteBuffer
import java.util.concurrent.ExecutorService
import java.util.concurrent.Executors

class MainActivity : ComponentActivity() {

    private lateinit var imageCapture: ImageCapture
    private lateinit var cameraExecutor: ExecutorService

    override fun onCreate(savedInstanceState: Bundle?) {
        super.onCreate(savedInstanceState)

        // Check camera permission
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
        val displayResults by remember { mutableStateOf("") }

        Column(modifier = Modifier.fillMaxSize(), verticalArrangement = Arrangement.SpaceBetween) {
            AndroidView(factory = { previewView }, modifier = Modifier
                .fillMaxWidth()
                .weight(1f))

            Button(
                onClick = {
                    takePhoto(context)
                },
                modifier = Modifier
                    .fillMaxWidth()
                    .padding(16.dp)
            ) {
                Text("Capture")
            }

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

    private fun takePhoto(context: android.content.Context) {
        val imageCapture = imageCapture ?: return

        imageCapture.takePicture(
            ContextCompat.getMainExecutor(context),
            object : ImageCapture.OnImageCapturedCallback() {
                override fun onCaptureSuccess(image: ImageProxy) {
                    val bitmap = imageProxyToBitmap(image)
                    Toast.makeText(context, "Captured!", Toast.LENGTH_SHORT).show()
                    // TODO: Feed bitmap into AI model
                    runObjectDetection(bitmap, context = context)
                    image.close()
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

    private fun runObjectDetection(bitmap: Bitmap, context: android.content.Context) {
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

        Log.e("ObjectDetection", "No ----objects detectedresults: $results")
        if (results.isEmpty()) {
            Toast.makeText(context, "No objects detected", Toast.LENGTH_SHORT).show()
            return
        }

        for (result in results) {
            val category = result.categories.firstOrNull()
            val label = category?.label ?: "Unknown"
            val score = category?.score ?: 0f

            Log.e("ObjectDetection", "Detected: $label (${"%.2f".format(score * 100)}%)")

            if (label.contains("bin", true) ||
                label.contains("trash", true)
                || label.contains("recycle", true)
                || label.contains("chair", true)
                || label.contains("toilet", true)) {
                Toast.makeText(context, "Object is detected: $label", Toast.LENGTH_SHORT).show()
            } else {
                Log.e("ObjectDetection", "Dustbin not detected")
            }
        }
    }

}
