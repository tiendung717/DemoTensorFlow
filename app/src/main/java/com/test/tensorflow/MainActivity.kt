package com.test.tensorflow

import android.content.Context
import android.graphics.*
import android.os.Bundle
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

import kotlin.math.min
import kotlin.math.max


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
                setTextColor(Color.WHITE)
                isChecked = mode == PorterDuff.Mode.DST_IN
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
        val ivMask = findViewById<ImageView>(R.id.ivMask)

        loadingView.visibility = View.VISIBLE
        ivResult.visibility = View.GONE
        ivMask.visibility = View.GONE

        lifecycleScope.launch(Dispatchers.IO) {
            val model = V3TrimmedQuantizationNoprepost.newInstance(context)

            val imageProcessor = ImageProcessor.Builder()
                .add(ResizeOp(SIZE, SIZE, ResizeOp.ResizeMethod.BILINEAR))
                .build()

            var tensorImage = TensorImage(DataType.FLOAT32)
            val inputBitmap = BitmapFactory.decodeResource(context.resources, image)

            tensorImage.load(inputBitmap)
            tensorImage = imageProcessor.process(tensorImage)

            // Creates inputs for reference.
            val input = TensorBuffer.createFixedSize(intArrayOf(1, 3, SIZE, SIZE), DataType.FLOAT32)
            input.loadBuffer(tensorImage.buffer)

            // Runs model inference and gets result.
            val outputs = model.process(input)
            val outputBuffer = outputs.outputFeature0AsTensorBuffer

            // Releases model resources if no longer used.
            model.close()
            
            val mask = convertToBitmap(
                floatArray = outputBuffer.floatArray,
                width = SIZE,
                height = SIZE,
                reverseScale = false
            )

            val final = saveCutout(tensorImage.bitmap, mask)


//  TEST ++
//            val sampleMask = BitmapFactory.decodeResource(context.resources, R.drawable.mask)
//            var tensorImageMask = TensorImage(DataType.FLOAT32)
//
//            tensorImageMask.load(sampleMask)
//            tensorImageMask = imageProcessor.process(tensorImageMask)
//
//            val resultBitmap = cutout(tensorImage.bitmap, tensorImageMask.bitmap)
//  TEST --

            withContext(Dispatchers.Main) {
                loadingView.visibility = View.GONE
                ivResult.visibility = View.VISIBLE
                ivMask.visibility = View.VISIBLE

                ivMask.setImageBitmap(mask)
                ivResult.setImageBitmap(final)
            }
        }

    }

    private fun saveCutout(sourceBitmap: Bitmap, mask: Bitmap): Bitmap {
        val result = Bitmap.createBitmap(SIZE, SIZE, Bitmap.Config.ARGB_8888)
        val canvas = Canvas(result)
        val paint = Paint(Paint.ANTI_ALIAS_FLAG)
        canvas.drawBitmap(sourceBitmap, 0f, 0f, paint)

        paint.xfermode = PorterDuffXfermode(PorterDuff.Mode.DST_IN)
        canvas.drawBitmap(mask, 0f, 0f, paint)

        paint.xfermode = null
        return result
    }

    private fun convertToBitmap(
        floatArray: FloatArray,
        width: Int,
        height: Int,
        alpha: Int = 255,
        reverseScale: Boolean = false
    ): Bitmap {
        val bitmap = Bitmap.createBitmap(width, height, Bitmap.Config.ARGB_8888)
        val conversion = when (reverseScale) {
            false -> { v: Float ->
                val p = max(min((v * 255).toInt(), 255), 0)
                p
            }
            true -> { v: Float ->
                val p = max(min((v * 255).toInt(), 255), 0)
                (255 - p)
            }
        }

        // copy each value from float array to RGB channels and set alpha channel
        for (h in 0 until height)
            for (w in 0 until width) {
                val i = width * h + w
                val value = conversion(floatArray[i])
                val pixel = Color.argb(alpha, value, value, value)
                bitmap.setPixel(w, h, pixel)
            }
        return bitmap
    }
    
    companion object {
        private const val SIZE = 320
    }
}