package edu.njucm.demo

import android.content.Intent
import android.support.v7.app.AppCompatActivity
import android.os.Bundle
import android.widget.Button



class MainActivity : AppCompatActivity() {
    override fun onCreate(savedInstanceState: Bundle?) {
        super.onCreate(savedInstanceState)
        setContentView(R.layout.activity_main)
        val btn1=findViewById<Button>(R.id.button)
        btn1.setOnClickListener{
            //var intent= Intent(this,Activity1::class.java)
            var intent=Intent(this,Activity1_1::class.java)
            startActivity(intent)


        }
        val btn2=findViewById<Button>(R.id.button2);
        btn2.setOnClickListener{
            var intent= Intent(this,Activity2_1::class.java)
            startActivity(intent)
            overridePendingTransition(android.R.anim.fade_in,android.R.anim.fade_out)

        }
        val btn3=findViewById<Button>(R.id.button3);
        btn3.setOnClickListener{
            var intent= Intent(this,Activity3_1::class.java)
            startActivity(intent)
            overridePendingTransition(android.R.anim.fade_in,android.R.anim.fade_out)

        }
        val btn5=findViewById<Button>(R.id.button5);
        btn5.setOnClickListener{
            var intent= Intent(this,Activity4_1::class.java)
            startActivity(intent)
            overridePendingTransition(android.R.anim.fade_in,android.R.anim.fade_out)
            finish()
        }
        val btn6=findViewById<Button>(R.id.button6);
        btn6.setOnClickListener{
            var intent= Intent(this,Activity5_1::class.java)
            startActivity(intent)
            overridePendingTransition(android.R.anim.fade_in,android.R.anim.fade_out)
            finish()
        }
        val btn7=findViewById<Button>(R.id.button7);
        btn7.setOnClickListener{
            var intent= Intent(this,Activity6::class.java)
            startActivity(intent)
            overridePendingTransition(android.R.anim.fade_in,android.R.anim.fade_out)
            finish()
        }
    }
}