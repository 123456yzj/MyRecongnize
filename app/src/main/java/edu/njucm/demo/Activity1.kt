package edu.njucm.demo

import android.content.Intent
import android.os.Bundle
import android.support.v7.app.AppCompatActivity
import android.widget.Button

class Activity1 :AppCompatActivity(){

    override fun onCreate(savedInstanceState: Bundle?) {
        super.onCreate(savedInstanceState)
        setContentView(R.layout.activity_1)
        val btn19=findViewById<Button>(R.id.button19);
        btn19.setOnClickListener(){
            var intent= Intent(this,MainActivity::class.java)
            startActivity(intent)
            finish()

        }
    }

}