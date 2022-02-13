package com.test.tensorflow

import android.content.Context
import android.graphics.Bitmap
import android.graphics.BitmapFactory
import android.graphics.Color
import android.os.Bundle
import android.util.Log
import android.view.View
import android.view.ViewGroup
import android.widget.ImageView
import androidx.annotation.DrawableRes
import androidx.appcompat.app.AppCompatActivity
import androidx.lifecycle.lifecycleScope
import androidx.viewpager.widget.PagerAdapter
import androidx.viewpager.widget.ViewPager
import com.test.tensorflow.ml.V3TrimmedQuantizationNoprepost
import kotlinx.coroutines.Dispatchers
import kotlinx.coroutines.launch
import kotlinx.coroutines.withContext
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

        initViews()
    }

    private fun initViews() {
        val vpImages = findViewById<ViewPager>(R.id.vpOriginImage)
        val btnRemoveBackground = findViewById<View>(R.id.btnRemoveBackground)

        val images = listOf<Int>(
            R.drawable.bike,
            R.drawable.boat,
            R.drawable.girl,
            R.drawable.hockey,
            R.drawable.horse,
            R.drawable.rifle1,
            R.drawable.rifle2,
            R.drawable.sailboat3,
            R.drawable.vangogh,
            R.drawable.whisk,
            R.drawable.im_01,
            R.drawable.im_14,
            R.drawable.im_21,
            R.drawable.im_27,
            R.drawable.lamp2_meitu_1
        )

        vpImages.adapter = ImagePagerAdapter(this, images)
        btnRemoveBackground.setOnClickListener {
            val currentImage = images[vpImages.currentItem]
            removeBackground(this, currentImage)
        }
    }

    private fun removeBackground(context: Context, @DrawableRes image: Int) {
        val loadingView = findViewById<View>(R.id.loadingView)
        val ivResult = findViewById<ImageView>(R.id.ivResult)

        loadingView.visibility = View.VISIBLE
        ivResult.visibility = View.GONE

        lifecycleScope.launch(Dispatchers.IO) {
            val model = V3TrimmedQuantizationNoprepost.newInstance(context)

            val imageProcessor = ImageProcessor.Builder()
                .add(ResizeOp(320, 320, ResizeOp.ResizeMethod.BILINEAR))
                .build()

            var tensorImage = TensorImage(DataType.FLOAT32)
            val inputBitmap = BitmapFactory.decodeResource(context.resources, image)

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

            val bitmap = getOutputImage(outputBuffer.buffer, 320, 320)

            withContext(Dispatchers.Main) {
                loadingView.visibility = View.GONE
                ivResult.visibility = View.VISIBLE
                ivResult.setImageBitmap(bitmap)
            }
        }

    }

    private fun getOutputImage(output: ByteBuffer, outputWidth: Int, outputHeight: Int): Bitmap {
        output.rewind() // Rewind the output buffer after running.
        output.order(ByteOrder.nativeOrder())

        val bitmap = Bitmap.createBitmap(outputWidth, outputHeight, Bitmap.Config.ARGB_8888)
        val pixels = IntArray(outputWidth * outputHeight) // Set your expected output's height and width
        for (i in 0 until outputWidth * outputHeight) {
            val a = 0xFF
            val r: Float = output.float * 255.0f
            val g: Float = output.float * 255.0f
            val b: Float = output.float * 255.0f
            pixels[i] = a shl 24 or (r.toInt() shl 16) or (g.toInt() shl 8) or b.toInt()
        }
        bitmap.setPixels(pixels, 0, outputWidth, 0, 0, outputWidth, outputHeight)

        return bitmap
    }
}