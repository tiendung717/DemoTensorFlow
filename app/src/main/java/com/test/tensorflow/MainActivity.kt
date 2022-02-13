package com.test.tensorflow

import android.content.Context
import android.graphics.Bitmap
import android.graphics.BitmapFactory
import android.os.Bundle
import android.util.Log
import android.widget.ImageView
import androidx.appcompat.app.AppCompatActivity
import com.test.tensorflow.ml.V3TrimmedQuantizationNoprepost
import org.tensorflow.lite.DataType
import org.tensorflow.lite.support.image.ImageProcessor
import org.tensorflow.lite.support.image.TensorImage
import org.tensorflow.lite.support.image.ops.ResizeOp
import org.tensorflow.lite.support.tensorbuffer.TensorBuffer


class MainActivity : AppCompatActivity() {
    override fun onCreate(savedInstanceState: Bundle?) {
        super.onCreate(savedInstanceState)
        setContentView(R.layout.activity_main)

        apply(this)
    }

    private fun apply(context: Context) {
        val model = V3TrimmedQuantizationNoprepost.newInstance(context)

        val imageProcessor = ImageProcessor.Builder()
            .add(ResizeOp(320, 320, ResizeOp.ResizeMethod.BILINEAR))
            .build()

        var tensorImage = TensorImage(DataType.FLOAT32)
        val inputBitmap = BitmapFactory.decodeResource(context.resources, R.drawable.boat)

        tensorImage.load(inputBitmap)
        tensorImage = imageProcessor.process(tensorImage)

        // Creates inputs for reference.
        val input = TensorBuffer.createFixedSize(intArrayOf(1, 3, 320, 320), DataType.FLOAT32)
        Log.d("nt.dung", "Size: ${input.flatSize}, buffer size: ${tensorImage.buffer.array().size}")
        input.loadBuffer(tensorImage.buffer)

        // Runs model inference and gets result.
        val outputs = model.process(input)
        val outputBuffer: TensorBuffer = outputs.outputFeature0AsTensorBuffer

        // Releases model resources if no longer used.
        model.close()


        val bitmap = BitmapFactory.decodeByteArray(outputBuffer.buffer.array(), 0, outputBuffer.buffer.array().size)
        findViewById<ImageView>(R.id.ivResult).setImageBitmap(bitmap)
    }
}