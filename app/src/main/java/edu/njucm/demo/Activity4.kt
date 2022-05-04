package edu.njucm.demo

import android.content.Intent
import android.os.Bundle
import android.support.v7.app.AppCompatActivity
import android.widget.Button

class Activity4 :AppCompatActivity(){
    override fun onCreate(savedInstanceState: Bundle?) {
        super.onCreate(savedInstanceState)
        setContentView(R.layout.activity_4)
        val btn8=findViewById<Button>(R.id.button8);
        btn8.setOnClickListener{
            var intent= Intent(this,MainActivity::class.java)
            startActivity(intent)
            overridePendingTransition(android.R.anim.fade_in,android.R.anim.fade_out)
            finish()
        }
        val btn9=findViewById<Button>(R.id.button9);
        btn9.setOnClickListener{
            var intent= Intent(this,Activity5_1::class.java)
            startActivity(intent)
            overridePendingTransition(android.R.anim.fade_in,android.R.anim.fade_out)
            finish()
        }
        val btn10=findViewById<Button>(R.id.button10);
        btn10.setOnClickListener{
            var intent= Intent(this,Activity6::class.java)
            startActivity(intent)
            overridePendingTransition(android.R.anim.fade_in,android.R.anim.fade_out)
            finish()
        }
    }
}