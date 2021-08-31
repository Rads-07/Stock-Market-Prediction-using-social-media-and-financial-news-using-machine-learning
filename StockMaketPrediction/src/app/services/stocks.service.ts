import { HttpClient } from '@angular/common/http';
import { Injectable } from '@angular/core';
import { environment } from 'src/environments/environment';

@Injectable({
  providedIn: 'root'
})
export class StocksService {

  
  constructor(private http:HttpClient) { }

  sendTicker(company:any){
    return this.http.post(environment.apiBaseUrl+'prediction/',company)
    //return {name:"hello"};
  }

}
