package edu.njucm.demo

import android.content.Intent
import android.os.Bundle
import android.support.v7.app.AppCompatActivity
import android.widget.Button

class Activity6 :AppCompatActivity(){
    override fun onCreate(savedInstanceState: Bundle?) {
        super.onCreate(savedInstanceState)
        setContentView(R.layout.activity_6)
        val btn14=findViewById<Button>(R.id.button14);
        btn14.setOnClickListener{
            var intent= Intent(this,MainActivity::class.java)
            startActivity(intent)
            overridePendingTransition(android.R.anim.fade_in,android.R.anim.fade_out)
            finish()
        }
        val btn15=findViewById<Button>(R.id.button15);
        btn15.setOnClickListener{
            var intent= Intent(this,Activity4_1::class.java)
            startActivity(intent)
            overridePendingTransition(android.R.anim.fade_in,android.R.anim.fade_out)
            finish()
        }
        val btn16=findViewById<Button>(R.id.button16);
        btn16.setOnClickListener{
            var intent= Intent(this,Activity5_1::class.java)
            startActivity(intent)
            overridePendingTransition(android.R.anim.fade_in,android.R.anim.fade_out)
            finish()
        }
        val btn21=findViewById<Button>(R.id.button21);
        btn21.setOnClickListener{
            var intent= Intent(this,LoginActivity2::class.java)
            startActivity(intent)
            overridePendingTransition(android.R.anim.fade_in,android.R.anim.fade_out)

        }
    }
}