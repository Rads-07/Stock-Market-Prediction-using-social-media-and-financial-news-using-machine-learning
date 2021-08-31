import { StocksService } from './../services/stocks.service';
import { Component, OnInit, ViewChild } from '@angular/core';
import { FormBuilder } from '@angular/forms';
import { MatSelect } from '@angular/material/select';
import { Router } from '@angular/router';


@Component({
  selector: 'app-prediction',
  templateUrl: './prediction.component.html',
  styleUrls: ['./prediction.component.scss']
})
export class PredictionComponent implements OnInit {

 
  constructor( private fb: FormBuilder, 
               private stockService:StocksService,
               private router: Router) { }

  companies:any[] = ['Reliance','TCS','HDFC ','Hindustan Unilever','Infosys','ICICI','Kotak Mahindra', 'Bajaj Finance', 'SBI']
  results:any;

   form = this.fb.group({
     company:[""]
     })

  ngOnInit():void {
  }

  async predict(company:any){
    let result = Object.values(company)
    console.log("company is "+result);
     this.results =  await this.stockService.sendTicker(result).toPromise();
      console.log(this.results)
      this.router.navigate(['result'],{ state:  {"result":this.results, "cname": result  } });
  }

}

