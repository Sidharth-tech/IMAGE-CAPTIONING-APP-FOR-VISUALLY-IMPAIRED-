package com.example.myapplication;

import androidx.activity.result.ActivityResultLauncher;
import androidx.activity.result.contract.ActivityResultContracts;
import androidx.annotation.NonNull;
import androidx.appcompat.app.AppCompatActivity;
import androidx.camera.core.Camera;
import androidx.camera.core.CameraSelector;
import androidx.camera.core.ImageCapture;
import androidx.camera.core.ImageCaptureException;
import androidx.camera.core.Preview;
import androidx.camera.lifecycle.ProcessCameraProvider;
import androidx.camera.view.PreviewView;
import androidx.core.app.ActivityCompat;
import androidx.core.content.ContextCompat;

import android.Manifest;
import android.content.Context;
import android.content.SharedPreferences;
import android.content.pm.PackageManager;
import android.graphics.Bitmap;
import android.graphics.BitmapFactory;
import android.os.Bundle;
import android.os.Handler;
import android.os.Looper;
import android.speech.tts.TextToSpeech;
import android.util.Base64;
import android.util.Log;
import android.view.View;
import android.widget.Button;
import android.widget.TextView;
import android.widget.Toast;

import com.google.common.util.concurrent.ListenableFuture;

import org.json.JSONException;
import org.json.JSONObject;

import java.io.ByteArrayOutputStream;
import java.io.File;
import java.io.IOException;
import java.util.Locale;
import java.util.concurrent.ExecutionException;
import java.util.concurrent.ExecutorService;
import java.util.concurrent.Executors;
import java.util.concurrent.TimeUnit;

import okhttp3.Call;
import okhttp3.Callback;
import okhttp3.MediaType;
import okhttp3.OkHttpClient;
import okhttp3.Request;
import okhttp3.RequestBody;
import okhttp3.Response;

public class MainActivity extends AppCompatActivity {
    private static final String TAG = "ImageCaptioning";
    private static final int REQUEST_CODE_PERMISSIONS = 10;
    private static final String[] REQUIRED_PERMISSIONS = new String[]{Manifest.permission.CAMERA};

    // UI Components
    private PreviewView previewView;
    private Button captureButton;
    private Button autoButton;
    private TextView captionTextView;

    // Camera variables
    private ExecutorService cameraExecutor;
    private ImageCapture imageCapture;

    // API variables
    private static final String API_URL = "http://192.168.216.32:5000/caption";
    private OkHttpClient client;

    // Text-to-speech
    private TextToSpeech textToSpeech;

    // Auto-capture
    private boolean autoCapture = false;
    private int captureInterval = 15; // seconds
    private Handler autoHandler;
    private Runnable autoRunnable;

    // Shared preferences
    private SharedPreferences preferences;
    private static final String PREFS_NAME = "ImageCaptioningPrefs";
    private static final String PREF_AUTO_CAPTURE = "auto_capture";
    private static final String PREF_CAPTURE_INTERVAL = "capture_interval";
    private static final String PREF_AUTO_SPEECH = "auto_speech";

    @Override
    protected void onCreate(Bundle savedInstanceState) {
        super.onCreate(savedInstanceState);
        setContentView(R.layout.activity_main);

        // Initialize UI components
        previewView = findViewById(R.id.preview_view);
        captureButton = findViewById(R.id.capture_button);
        autoButton = findViewById(R.id.auto_button);
        captionTextView = findViewById(R.id.caption_text);

        // Load preferences
        preferences = getSharedPreferences(PREFS_NAME, Context.MODE_PRIVATE);
        autoCapture = preferences.getBoolean(PREF_AUTO_CAPTURE, false);
        captureInterval = preferences.getInt(PREF_CAPTURE_INTERVAL, 15);
        boolean autoSpeech = preferences.getBoolean(PREF_AUTO_SPEECH, true);

        // Initialize camera executor
        cameraExecutor = Executors.newSingleThreadExecutor();

        // Initialize OkHttp client
        client = new OkHttpClient.Builder()
                .connectTimeout(30, TimeUnit.SECONDS)
                .writeTimeout(30, TimeUnit.SECONDS)
                .readTimeout(30, TimeUnit.SECONDS)
                .build();

        // Initialize text-to-speech
        textToSpeech = new TextToSpeech(this, status -> {
            if (status != TextToSpeech.ERROR) {
                textToSpeech.setLanguage(Locale.US);
            }
        });

        // Initialize auto capture handler
        autoHandler = new Handler(Looper.getMainLooper());
        autoRunnable = new Runnable() {
            @Override
            public void run() {
                if (autoCapture) {
                    captureImage();
                    autoHandler.postDelayed(this, captureInterval * 1500);
                }
            }
        };

        // Set up capture button
        captureButton.setOnClickListener(v -> captureImage());

        // Set up auto button
        updateAutoButtonText();
        autoButton.setOnClickListener(v -> toggleAutoCapture());

        // Check permissions
        if (allPermissionsGranted()) {
            startCamera();
        } else {
            ActivityCompat.requestPermissions(this, REQUIRED_PERMISSIONS, REQUEST_CODE_PERMISSIONS);
        }
    }

    private void startCamera() {
        ListenableFuture<ProcessCameraProvider> cameraProviderFuture =
                ProcessCameraProvider.getInstance(this);

        cameraProviderFuture.addListener(() -> {
            try {
                ProcessCameraProvider cameraProvider = cameraProviderFuture.get();

                // Set up preview
                Preview preview = new Preview.Builder().build();
                preview.setSurfaceProvider(previewView.getSurfaceProvider());

                // Set up image capture
                imageCapture = new ImageCapture.Builder()
                        .setCaptureMode(ImageCapture.CAPTURE_MODE_MINIMIZE_LATENCY)
                        .build();

                // Select back camera
                CameraSelector cameraSelector = new CameraSelector.Builder()
                        .requireLensFacing(CameraSelector.LENS_FACING_BACK)
                        .build();

                // Unbind any bound use cases before rebinding
                cameraProvider.unbindAll();

                // Bind use cases to camera
                Camera camera = cameraProvider.bindToLifecycle(
                        this, cameraSelector, preview, imageCapture);

                // Enable capture button
                captureButton.setEnabled(true);
                autoButton.setEnabled(true);

                // Start auto capture if enabled
                if (autoCapture) {
                    startAutoCapture();
                }

            } catch (ExecutionException | InterruptedException e) {
                Log.e(TAG, "Error starting camera: " + e.getMessage());
                Toast.makeText(this, "Error starting camera", Toast.LENGTH_SHORT).show();
            }
        }, ContextCompat.getMainExecutor(this));
    }

    private void captureImage() {
        if (imageCapture == null) {
            Toast.makeText(this, "Camera not ready", Toast.LENGTH_SHORT).show();
            return;
        }

        // Show processing message
        captionTextView.setText("Processing image...");

        // Create temp file for image
        File outputDir = getApplicationContext().getCacheDir();
        File outputFile;
        try {
            outputFile = File.createTempFile("captured_image", ".jpg", outputDir);
        } catch (IOException e) {
            Log.e(TAG, "Error creating temp file: " + e.getMessage());
            Toast.makeText(this, "Failed to create temp file", Toast.LENGTH_SHORT).show();
            return;
        }

        // Output options
        ImageCapture.OutputFileOptions outputOptions =
                new ImageCapture.OutputFileOptions.Builder(outputFile).build();

        // Take picture
        imageCapture.takePicture(
                outputOptions,
                ContextCompat.getMainExecutor(this),
                new ImageCapture.OnImageSavedCallback() {
                    @Override
                    public void onImageSaved(@NonNull ImageCapture.OutputFileResults outputFileResults) {
                        processImage(outputFile);
                    }

                    @Override
                    public void onError(@NonNull ImageCaptureException exception) {
                        Log.e(TAG, "Image capture failed: " + exception.getMessage());
                        Toast.makeText(MainActivity.this,
                                "Image capture failed", Toast.LENGTH_SHORT).show();
                        captionTextView.setText("Image capture failed");
                    }
                });
    }

    private void processImage(File imageFile) {
        // Convert image to base64 string
        Bitmap bitmap = BitmapFactory.decodeFile(imageFile.getAbsolutePath());
        ByteArrayOutputStream byteArrayOutputStream = new ByteArrayOutputStream();
        bitmap.compress(Bitmap.CompressFormat.JPEG, 80, byteArrayOutputStream);
        byte[] byteArray = byteArrayOutputStream.toByteArray();
        String base64Image = Base64.encodeToString(byteArray, Base64.DEFAULT);

        // Create JSON request
        JSONObject jsonBody = new JSONObject();
        try {
            jsonBody.put("image", base64Image);
        } catch (JSONException e) {
            Log.e(TAG, "Error creating JSON: " + e.getMessage());
            return;
        }

        // Create request
        MediaType JSON = MediaType.parse("application/json; charset=utf-8");
        RequestBody body = RequestBody.create(jsonBody.toString(), JSON);
        Request request = new Request.Builder()
                .url(API_URL)
                .post(body)
                .build();

        // Send request
        client.newCall(request).enqueue(new Callback() {
            @Override
            public void onFailure(@NonNull Call call, @NonNull IOException e) {
                Log.e(TAG, "API request failed: " + e.getMessage());
                runOnUiThread(() -> {
                    Toast.makeText(MainActivity.this,
                            "Failed to connect to server", Toast.LENGTH_SHORT).show();
                    captionTextView.setText("Failed to connect to server");
                });
            }

            @Override
            public void onResponse(@NonNull Call call, @NonNull Response response) throws IOException {
                if (!response.isSuccessful()) {
                    Log.e(TAG, "API error: " + response.code());
                    runOnUiThread(() -> {
                        Toast.makeText(MainActivity.this,
                                "Server error: " + response.code(), Toast.LENGTH_SHORT).show();
                        captionTextView.setText("Server error: " + response.code());
                    });
                    return;
                }

                // Process response
                try {
                    String responseBody = response.body().string();
                    JSONObject jsonResponse = new JSONObject(responseBody);
                    String caption = jsonResponse.getString("caption");

                    // Update UI
                    runOnUiThread(() -> {
                        captionTextView.setText(caption);

                        // Speak caption if enabled
                        boolean autoSpeech = preferences.getBoolean(PREF_AUTO_SPEECH, true);
                        if (autoSpeech) {
                            speakText(caption);
                        }
                    });
                } catch (JSONException e) {
                    Log.e(TAG, "Error parsing JSON response: " + e.getMessage());
                    runOnUiThread(() -> {
                        Toast.makeText(MainActivity.this,
                                "Error parsing server response", Toast.LENGTH_SHORT).show();
                        captionTextView.setText("Error parsing server response");
                    });
                }
            }
        });
    }

    private void speakText(String text) {
        if (textToSpeech != null) {
            textToSpeech.speak(text, TextToSpeech.QUEUE_FLUSH, null, null);
        }
    }

    private void toggleAutoCapture() {
        autoCapture = !autoCapture;

        // Update preferences
        SharedPreferences.Editor editor = preferences.edit();
        editor.putBoolean(PREF_AUTO_CAPTURE, autoCapture);
        editor.apply();

        // Update UI
        updateAutoButtonText();

        if (autoCapture) {
            startAutoCapture();
        } else {
            stopAutoCapture();
        }
    }

    private void updateAutoButtonText() {
        if (autoCapture) {
            autoButton.setText("Auto: ON (" + captureInterval + "s)");
        } else {
            autoButton.setText("Auto: OFF");
        }
    }

    private void startAutoCapture() {
        autoHandler.postDelayed(autoRunnable, captureInterval * 1500);
    }

    private void stopAutoCapture() {
        autoHandler.removeCallbacks(autoRunnable);
    }

    private boolean allPermissionsGranted() {
        for (String permission : REQUIRED_PERMISSIONS) {
            if (ContextCompat.checkSelfPermission(this, permission) !=
                    PackageManager.PERMISSION_GRANTED) {
                return false;
            }
        }
        return true;
    }

    @Override
    public void onRequestPermissionsResult(int requestCode, @NonNull String[] permissions,
                                           @NonNull int[] grantResults) {
        super.onRequestPermissionsResult(requestCode, permissions, grantResults);
        if (requestCode == REQUEST_CODE_PERMISSIONS) {
            if (allPermissionsGranted()) {
                startCamera();
            } else {
                Toast.makeText(this,
                        "Permissions not granted by the user.", Toast.LENGTH_SHORT).show();
                finish();
            }
        }
    }

    @Override
    protected void onResume() {
        super.onResume();
        if (autoCapture) {
            startAutoCapture();
        }
    }

    @Override
    protected void onPause() {
        super.onPause();
        stopAutoCapture();
    }

    @Override
    protected void onDestroy() {
        super.onDestroy();
        cameraExecutor.shutdown();
        if (textToSpeech != null) {
            textToSpeech.stop();
            textToSpeech.shutdown();
        }
    }
}