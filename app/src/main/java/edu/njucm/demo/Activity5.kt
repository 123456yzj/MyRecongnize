package edu.njucm.demo

import android.content.Intent
import android.os.Bundle
import android.support.v7.app.AppCompatActivity
import android.widget.Button

class Activity5:AppCompatActivity() {
    override fun onCreate(savedInstanceState: Bundle?) {
        super.onCreate(savedInstanceState)
        setContentView(R.layout.activity_5)
        val btn11=findViewById<Button>(R.id.button11);
        btn11.setOnClickListener{
            var intent= Intent(this,MainActivity::class.java)
            startActivity(intent)
            overridePendingTransition(android.R.anim.fade_in,android.R.anim.fade_out)
            finish()
        }
        val btn12=findViewById<Button>(R.id.button12);
        btn12.setOnClickListener{
            var intent= Intent(this,Activity4::class.java)
            startActivity(intent)
            overridePendingTransition(android.R.anim.fade_in,android.R.anim.fade_out)
            finish()
        }
        val btn13=findViewById<Button>(R.id.button13);
        btn13.setOnClickListener{
            var intent= Intent(this,Activity6::class.java)
            startActivity(intent)
            overridePendingTransition(android.R.anim.fade_in,android.R.anim.fade_out)
            finish()
        }
    }
}