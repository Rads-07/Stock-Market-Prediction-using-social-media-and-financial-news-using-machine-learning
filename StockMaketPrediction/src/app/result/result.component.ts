import { Component, OnInit } from '@angular/core';
import { Router } from '@angular/router';

declare let google: any;
@Component({
  selector: 'app-result',
  templateUrl: './result.component.html',
  styleUrls: ['./result.component.scss']
})
export class ResultComponent implements OnInit {

  constructor( private router:Router) { }
  public output :any
  public prediction:any
  public df1 !:any[]
  public df2 !:any[]
  public df3 !:any[]
  public df4 !:any[]
  public cname:any

  async ngOnInit(){
    let temp = window.history.state["result"];
    console.log(temp);
    this.cname =  window.history.state["cname"];
    this.output  = temp["result"]
    
    this.df1 = temp["df1"]
    this.df2 = temp["df2"]
    this.df3 = temp["pie1"]
    this.df4 = temp["pie2"]
    this.prediction = temp["prediction"]
   

    await google.charts.load('current', { packages: ['corechart','line'] });
    google.charts.setOnLoadCallback(this.drawChart1());
    google.charts.setOnLoadCallback(this.drawChart2());
    google.charts.setOnLoadCallback(this.drawChart3());
    google.charts.setOnLoadCallback(this.drawChart4());

  }

  drawChart1() {
    var data = new google.visualization.DataTable([]);

     data.addColumn('string', 'Date');
     data.addColumn('number', 'close');

     data.addRows(this.df1);
     var options = {
      title: 'Closing Price',
      curveType: 'function',
        legend: {
            position: 'bottom',
        },
        chartArea:{right:40,top:50,left:50, bottom:100}
      }
    var chart = new google.visualization.LineChart(document.getElementById('curve_chart'));

    chart.draw(data, options);
  
  }
  
  drawChart2()   //for candle stick of open,close,low,high
  {
    var data = google.visualization.arrayToDataTable(
     this.df2, true);

    const options ={
      title: 'Candlestick Analysis of Stock Prices',
      legend:'none',
      Animation: 30,
      chartArea:{right:40,top:50,left:50, bottom:50}
    }
    const chart = new google.visualization.CandlestickChart( document.querySelector('#chart_div'));
    chart.draw(data, options);
  }

  drawChart3() {

    var data = new google.visualization.DataTable([]);
    data.addColumn('string','Tweet_type')
    data.addColumn('number','Count')
    data.addRows(this.df3);

    var options = {
      title: 'Twitter Sentiment',
      is3D: true,
      chartArea:{right:20,top:50,left:20, bottom:10},
      slices: {
        1: {color: '#0D47A1'},
        0: {color: '#64B5F6'},
      },
    };

    var chart = new google.visualization.PieChart(document.getElementById('piechart1'));

    chart.draw(data, options);
  }

  drawChart4() {

    var data = new google.visualization.DataTable([]);
    data.addColumn('string','News_type')
    data.addColumn('number','Count')
    data.addRows(this.df4);

    var options = {
      title: 'Financial News Sentiment',
      is3D: true,
      chartArea:{right:20,top:50,left:20, bottom:10},
      slices: {
        0: {color: '#64B5F6'},
        1: {color: '#0D47A1'},
      }
    };

    var chart = new google.visualization.PieChart(document.getElementById('piechart2'));
    chart.draw(data, options);
    
  }
}
