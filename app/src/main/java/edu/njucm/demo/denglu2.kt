package edu.njucm.demo

import android.content.Intent
import android.os.Bundle
import android.support.v7.app.AppCompatActivity
import android.widget.Button

class denglu2 :AppCompatActivity(){
    override fun onCreate(savedInstanceState: Bundle?) {
        super.onCreate(savedInstanceState)
        setContentView(R.layout.activitydenglu)
        val btn1=findViewById<Button>(R.id.button);
        btn1.setOnClickListener{
            var intent= Intent(this,LoginActivity2::class.java)
            startActivity(intent)

        }
        val btn2=findViewById<Button>(R.id.button2);
        btn2.setOnClickListener{
            var intent= Intent(this,RegisterActivity::class.java)
            startActivity(intent)

        }
    }
}