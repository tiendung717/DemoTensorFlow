package com.test.tensorflow

import android.content.Context
import android.graphics.Bitmap
import android.graphics.BitmapFactory
import android.graphics.Color
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
import java.nio.ByteBuffer
import java.nio.ByteOrder


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
        val inputBitmap = BitmapFactory.decodeResource(context.resources, R.drawable.bike)

        tensorImage.load(inputBitmap)
        tensorImage = imageProcessor.process(tensorImage)

        // Creates inputs for reference.
        val input = TensorBuffer.createFixedSize(intArrayOf(1, 3, 320, 320), DataType.FLOAT32)
        input.loadBuffer(tensorImage.buffer)

        // Runs model inference and gets result.
        val outputs = model.process(input)
        val outputBuffer: TensorBuffer = outputs.outputFeature0AsTensorBuffer

        // Releases model resources if no longer used.
        model.close()

        val bitmap = convertByteBufferToBitmap(outputBuffer.buffer, 320, 320)
        findViewById<ImageView>(R.id.ivResult).setImageBitmap(bitmap)
    }

    /**
     * Converts ByteBuffer with segmentation mask to the Bitmap
     *
     * @param byteBuffer Output ByteBuffer from Interpreter.run
     * @param imgSizeX Model output image width
     * @param imgSizeY Model output image height
     * @return Mono color Bitmap mask
     */
    private fun convertByteBufferToBitmap(
        byteBuffer: ByteBuffer,
        imgSizeX: Int,
        imgSizeY: Int
    ): Bitmap? {
        byteBuffer.rewind()
        byteBuffer.order(ByteOrder.nativeOrder())
        val bitmap = Bitmap.createBitmap(imgSizeX, imgSizeY, Bitmap.Config.ARGB_8888)
        val pixels = IntArray(imgSizeX * imgSizeY)
        for (i in 0 until imgSizeX * imgSizeY) if (byteBuffer.float > 0.5) pixels[i] =
            Color.argb(255, 255, 105, 255) else pixels[i] = Color.argb(0, 0, 0, 0)
        bitmap.setPixels(pixels, 0, imgSizeX, 0, 0, imgSizeX, imgSizeY)
        return bitmap
    }
}