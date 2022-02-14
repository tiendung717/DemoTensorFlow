package com.test.tensorflow

import android.content.Context
import android.graphics.*
import android.os.Bundle
import android.util.Log
import android.view.View
import android.widget.ImageView
import android.widget.RadioButton
import android.widget.RadioGroup
import androidx.annotation.DrawableRes
import androidx.appcompat.app.AppCompatActivity
import androidx.lifecycle.lifecycleScope
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

    private val mode = listOf(
        PorterDuff.Mode.SRC_IN,
        PorterDuff.Mode.SRC_OUT,
        PorterDuff.Mode.SRC_OVER,
        PorterDuff.Mode.SRC_ATOP,
        PorterDuff.Mode.DST_IN,
        PorterDuff.Mode.DST_OUT,
        PorterDuff.Mode.DST_ATOP,
        PorterDuff.Mode.DST_OVER,
    )

    private val rdoGroup by lazy { findViewById<RadioGroup>(R.id.rdoGroup) }

    override fun onCreate(savedInstanceState: Bundle?) {
        super.onCreate(savedInstanceState)
        setContentView(R.layout.activity_main)

        initViews()
    }

    private fun initViews() {
        val rdoGroup = findViewById<RadioGroup>(R.id.rdoGroup)
        mode.forEachIndexed { index, mode ->
            val rdo = RadioButton(this).apply {
                id = index
                text = mode.name
            }
            rdoGroup.addView(rdo)
        }

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

            val bitmapMask = floatArrayToGrayscaleBitmap(
                floatArray = outputBuffer.floatArray,
                width = 320,
                height = 320,
                reverseScale = true
            )

            val resultBitmap = cutout(tensorImage.bitmap, bitmapMask)

            withContext(Dispatchers.Main) {
                loadingView.visibility = View.GONE
                ivResult.visibility = View.VISIBLE
                ivResult.setImageBitmap(resultBitmap)
            }
        }

    }

    private fun cutout(sourceBitmap: Bitmap, mask: Bitmap) : Bitmap {
        val resultBitmap = Bitmap.createBitmap(320, 320, Bitmap.Config.ARGB_8888)
        val canvas = Canvas(resultBitmap)
        canvas.drawBitmap(sourceBitmap, 0f, 0f, null)

        val paint = Paint().apply {
            xfermode = PorterDuffXfermode(mode[rdoGroup.checkedRadioButtonId])
        }
        canvas.drawBitmap(mask, 0f, 0f, paint)
        return resultBitmap
    }

    private fun getOutputImage(
        output: ByteBuffer,
        width: Int,
        height: Int
    ): Bitmap {
        output.rewind()
        output.order(ByteOrder.nativeOrder())
        val bitmap = Bitmap.createBitmap(width, height, Bitmap.Config.ARGB_8888)
        val pixels = IntArray(width * height * 4) // Set your expected output's height and width

        for (h in 0..height)
            for (w in 0..width) {
                val i = (width * h + w)
                val a = 0xFF
                val r: Float = output.getFloat(width * h + w) * 255.0f
                val g: Float = output.getFloat(width * h + w) * 255.0f
                val b: Float = output.getFloat(width * h + w) * 255.0f
                Log.d("nt.dung", "r: $r, g: $g, b: $b")

                pixels[i] = a shl 24 or (r.toInt() shl 16) or (g.toInt() shl 8) or b.toInt()
            }

        bitmap.setPixels(pixels, 0, width, 0, 0, width, height)

        return bitmap
    }

    private fun floatArrayToGrayscaleBitmap(
        floatArray: FloatArray,
        width: Int,
        height: Int,
        alpha: Byte = (255).toByte(),
        reverseScale: Boolean = false
    ): Bitmap {

        // Create empty bitmap in RGBA format (even though it says ARGB but channels are RGBA)
        val bmp = Bitmap.createBitmap(width, height, Bitmap.Config.ARGB_8888)
        val byteBuffer = ByteBuffer.allocate(width * height * 4)

        // mapping smallest value to 0 and largest value to 255
        val maxValue = floatArray.maxOrNull() ?: 1.0f
        val minValue = floatArray.minOrNull() ?: 0.0f
        val delta = maxValue - minValue
        var tempValue: Byte

        // Define if float min..max will be mapped to 0..255 or 255..0
        val conversion = when (reverseScale) {
            false -> { v: Float -> ((v - minValue) / delta * 255).toInt().toByte() }
            true -> { v: Float -> (255 - (v - minValue) / delta * 255).toInt().toByte() }
        }

        // copy each value from float array to RGB channels and set alpha channel
        floatArray.forEachIndexed { i, value ->
            tempValue = conversion(value)
            byteBuffer.put(4 * i, tempValue) // r
            byteBuffer.put(4 * i + 1, tempValue) // g
            byteBuffer.put(4 * i + 2, tempValue) // b
            byteBuffer.put(4 * i + 3, alpha) // a
        }

        bmp.copyPixelsFromBuffer(byteBuffer)

        return bmp
    }
}