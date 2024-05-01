import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import matplotlib.pyplot as plt
from datetime import timedelta
metric_title_map = {
    "abs_error": "MAE",
    
    "accuracy": "Forecasting Accuracy",
}



def prepare_summary_reports(pred):
    fig = go.Figure(
        make_subplots(
            rows=2,
            cols=1,
            shared_xaxes=True,
            vertical_spacing=0.1,
            specs=[[{"type": "table"}], [{"type": "table"}]],
            subplot_titles=("Report 1",  "Report 2"),
        )
    )
    names = {}
    plot_index = 0
    row = 0
    df_group_dict = {}
    for metric in ["abs_error",  "accuracy"]:
        row += 1
        plot_index += 1
        error_cols = [col for col in pred.columns if metric in col]
        df_group = pred[error_cols].describe()
        df_group = df_group.round(decimals=4)
        df_group.reset_index(inplace=True)
        df_group_dict[f"df_group_{plot_index}"] = df_group

        fig = fig.add_trace(
            go.Table(
                header=dict(
                    values=list(df_group.columns), font=dict(size=10), align="left"
                ),
                cells=dict(
                    values=[df_group[k].tolist() for k in df_group.columns[0:]],
                    align="left",
                ),
            ),
            row=row,
            col=1,
        )
        names.update({f"Report {plot_index}": f"{metric_title_map[metric]} summary"})
    fig.update_layout(title=f"Forecasting Summary")
    fig.for_each_annotation(lambda a: a.update(text=a.text + ": " + names[a.text]))
    return df_group_dict, fig


def metric_mean_plot(pred, metric, period):
    fig = go.Figure(
        make_subplots(
            rows=2, cols=1, shared_xaxes=False, subplot_titles=("Plot 1", "Plot 2")
        )
    )
    row = 0
    names = {}
    for side in ["high", "low"]:
        row += 1
        for quantile_type in ["q5", "q40", "q50", "q60", "q95"]:
            if f"{side}_{quantile_type}_{metric}" in pred.columns:
                data = pred[f"{side}_{quantile_type}_{metric}"].resample(period).agg("mean")
                if not data.empty:
                    fig.add_trace(
                        go.Scatter(
                            x=data
                            .index,
                            y=data,
                            name=f"{side} 7 days MAE: {quantile_type}",
                            opacity=0.5,
                            mode="lines",
                        ),
                        row=row,
                        col=1,
                    )
                    names.update(
                        {f"Plot {row}": f"{side} {metric_title_map[metric]} (window = {period})"}
                    )

    fig.update_layout(title=f"{metric_title_map[metric]} Plots")
    fig.for_each_annotation(lambda a: a.update(text=a.text + ": " + names[a.text]))
    return fig


def resduial_error(pred, side):
    fig = px.scatter(
        pred,
        x=pred.index,
        y=f"{side}_q50_error",
        marginal_y="violin",
        trendline="ols",
        trendline_color_override="#444",
    )
    fig.update_layout(title=f"{side} Resduial Error")
    return fig


def pips_histogram(pred, side):
    x0 = pred[f"{side}_q50_pips"]
    x1 = pred[f"{side}_actual_pips"]

    df = pd.DataFrame(
        dict(
            series=np.concatenate(
                (
                    [f"{side}_predicted_pips"] * len(x0),
                    [f"{side}_actual_pips"] * len(x1),
                )
            ),
            data=np.concatenate((x0, x1)),
        )
    )

    fig = px.histogram(df, x="data", color="series", barmode="overlay")
    fig.update_layout(title=f"{side}: Actual Vs Estimated Pips from open price")
    return fig


def plot_estimated_prices_range(pred, side, year):
    kind = "bid" if side == "high" else "ask"
    opposite_kind = "ask" if side == "bid" else "ask"
    fig = go.Figure(
        [
            go.Scatter(
                x=pred[f"{side}_q50"].index,
                y=pred[f"{side}_q50"],
                name=f"Estimated {side} price",
                opacity=1,
                mode="lines",
            ),
            go.Scatter(
                name=f"Estimated {side} Lower Bound",
                x=pred.index,
                y=pred[f"{side}_q5"],
                marker=dict(color="#444"),
                line=dict(width=0.25),
                mode="lines",
                fillcolor="rgba(68, 68, 68, 0.3)",
                fill="tonexty",
                showlegend=True,
            ),
            go.Scatter(
                name=f"Estimated {side} Upper Bound",
                x=pred.index,
                y=pred[f"{side}_q95"],
                marker=dict(color="#444"),
                line=dict(width=0.25),
                mode="lines",
                fillcolor="rgba(68, 68, 68, 0.3)",
                fill="tonexty",
                showlegend=True,
            ),
            go.Scatter(
                x=pred[f"{side}_{kind}"].index,
                y=pred[f"{side}_{kind}"],
                name=f"Actaul {side} price",
                opacity=1,
                mode="lines",
            ),
            go.Scatter(
                x=pred[f"open_{opposite_kind}"].index,
                y=pred[f"open_{opposite_kind}"],
                name=f"Actaul open price",
                opacity=1,
                mode="lines",
            ),
        ]
        + [
            go.Scatter(
                x=pred[f"{side}_{quantile_type}"].index,
                y=pred[f"{side}_{quantile_type}"],
                name=f"Estimated {side}: {quantile_type}",
                opacity=0.75,
                line=dict(dash="dash"),
            )
            for quantile_type in ["q40", "q60"]
        ]
    )
    fig.update_layout(title=f"{side} Estimated vs Actual Prices Plot (Year = {year})")
    fig = fig.update_xaxes(rangebreaks=[dict(bounds=["sat", "sun"])])  # hide weekends
    return fig

def plot_candlestick(pred, year):
    fig = go.Figure()

    candlestick_trace = go.Candlestick(
        x=pred.index,
        open=pred["open_avg"],
        high=pred["high_bid"],
        low=pred["low_ask"],
        close=pred["close_avg"],
    )
    fig.add_trace(candlestick_trace)

    trace_names = {
        "high_q50": "Estimated High",
        "high_q95": "Estimated High Upper Bound",
        "high_q5": "Estimated High Lower Bound",
        "low_q50": "Estimated Low",
        "low_q95": "Estimated Low Upper Bound",
        "low_q5": "Estimated Low Lower Bound",
    }

    for trace_name, trace_title in trace_names.items():
        if trace_name in pred.columns:
            trace = go.Scatter(
                name=trace_title,
                x=pred.index,
                y=pred[trace_name],
                mode="lines",
                line=dict(color="rgb(31, 119, 180)", width=1),
            )
            fig.add_trace(trace)

            if trace_name in ["high_q95", "high_q5", "low_q95", "low_q5"]:
                fill_trace = go.Scatter(
                    x=pred.index,
                    y=pred[trace_name],
                    mode="lines",
                    line=dict(width=0),
                    fillcolor="rgba(68, 68, 68, 0.3)",
                    fill="tonexty",
                    showlegend=False,
                )
                fig.add_trace(fill_trace)

    fig.update_layout(title=f"Estimated vs Actual CandleSticks Plot (Year == {year})")
    fig.update_xaxes(rangebreaks=[dict(bounds=["sat", "sun"])])  # hide weekends
    fig.update_yaxes(fixedrange=False)
    return fig
# def plot_candlestick(pred, year):
#     fig = go.Figure(
#         [
#             go.Candlestick(
#                 x=pred.index,
#                 open=pred["open_avg"],
#                 high=pred["high_bid"],
#                 low=pred["low_ask"],
#                 close=pred["close_avg"],
#             ),
#             go.Scatter(
#                 name="Estimated High",
#                 x=pred.index,
#                 y=pred["high_q50"],
#                 mode="lines",
#                 line=dict(color="rgb(31, 119, 180)", width=1),
#             ),
#             go.Scatter(
#                 name="Estimated High Upper Bound",
#                 x=pred.index,
#                 y=pred["high_q95"],
#                 marker=dict(color="#444"),
#                 line=dict(width=0),
#                 mode="lines",
#                 fillcolor="rgba(68, 68, 68, 0.3)",
#                 fill="tonexty",
#                 showlegend=False,
#             ),
#             go.Scatter(
#                 name="Estimated High Lower Bound",
#                 x=pred.index,
#                 y=pred["high_q5"],
#                 marker=dict(color="#444"),
#                 line=dict(width=0),
#                 mode="lines",
#                 fillcolor="rgba(68, 68, 68, 0.3)",
#                 fill="tonexty",
#                 showlegend=False,
#             ),
#             go.Scatter(
#                 name="Estimated Low",
#                 x=pred.index,
#                 y=pred["low_q50"],
#                 mode="lines",
#                 line=dict(color="rgb(31, 119, 180)", width=1),
#             ),
#             go.Scatter(
#                 name="Estimated Low Upper Bound",
#                 x=pred.index,
#                 y=pred["low_q95"],
#                 marker=dict(color="#444"),
#                 line=dict(width=0),
#                 mode="lines",
#                 fillcolor="rgba(68, 68, 68, 0.3)",
#                 fill="tonexty",
#                 showlegend=False,
#             ),
#             go.Scatter(
#                 name="Estimated Low Lower Bound",
#                 x=pred.index,
#                 y=pred["low_q5"],
#                 marker=dict(color="#444"),
#                 line=dict(width=0),
#                 mode="lines",
#                 fillcolor="rgba(68, 68, 68, 0.3)",
#                 fill="tonexty",
#                 showlegend=False,
#             ),
#         ]
#     )

#     fig.update_layout(title=f"Estimated vs Actual CandleSticks Plot (Year == {year})")

#     fig = fig.update_xaxes(rangebreaks=[dict(bounds=["sat", "sun"])])  # hide weekends

#     fig.update_yaxes(fixedrange=False)
#     return fig



def plot_comparison(decoded_data,orginal_test_data,threshold,column_name='high_bid'):
    plt.figure(figsize=(10, 6))

    denoised_difference = decoded_data[column_name] - orginal_test_data[column_name]
    denoised_abnormal = abs(denoised_difference) > threshold

   
    denoised_abnormal = denoised_abnormal[decoded_data.index]

    plt.plot(decoded_data.index, decoded_data[column_name], label='Decoded Data', color='blue')
    plt.plot(orginal_test_data.index, orginal_test_data[column_name], label='Original Test Data', color='green')
    plt.scatter(decoded_data.index[denoised_abnormal], decoded_data[column_name][denoised_abnormal], color='black', label='denoised_Abnormal')
    plt.xlabel('Datetime')
    plt.ylabel(column_name)
    plt.title('High Bid Comparison')
    plt.legend()
    plt.grid(True)
    plt.show()   

def plot_abnormal_period(predictions, orginal_test_data, threshold):
    pred_difference = predictions['high_bid'] - orginal_test_data['high_bid'] 

    pred_abnormal = abs(pred_difference) > threshold
    pred_abnormal_indices = pred_abnormal[pred_abnormal].index
    plt.figure(figsize=(10, 6))

    plt.plot(orginal_test_data.index, orginal_test_data['high_bid'], label='Original Test Data', color='green')

    time_diff = pred_abnormal_indices.to_series().diff()

    abnormal_period_start = None
    abnormal_period_count = 0
    abnormal_periods = []

    for index, time_diff_value in time_diff.items():
        if time_diff_value == pd.Timedelta(hours=1):
            if abnormal_period_start is None:
                abnormal_period_start = index
            abnormal_period_count += 1
            if abnormal_period_count >= 5:
                abnormal_periods.append((abnormal_period_start, abnormal_period_count))
        else:
            abnormal_period_start = None
            abnormal_period_count = 0

    for start_index, period_length in abnormal_periods:
        end_index = start_index + pd.Timedelta(hours=period_length)
        selected_data = orginal_test_data.loc[start_index:end_index]
        plt.scatter(selected_data.index, selected_data['high_bid'], color='red')

    print(abnormal_periods)
    
    plt.xlabel('Datetime')
    plt.ylabel('High Bid')
    plt.title('High Bid Comparison with Abnormal Periods')
    plt.legend()
    plt.grid(True)
    plt.show()

def check_overlap(predictions, decoded_data, orginal_test_data, threshold,column_name):
    
    pred_abnormal_indices = check_abnormal_indices(predictions, orginal_test_data, threshold,column_name)

    
    denoised_abnormal_indices = check_abnormal_indices(decoded_data, orginal_test_data, threshold,column_name)

    index_overlap = pred_abnormal_indices.intersection(denoised_abnormal_indices)

    print("Index Overlap:", index_overlap)

def check_abnormal_indices(output_data, orginal_test_data, threshold,column_name):
    difference = output_data[column_name] - orginal_test_data[column_name]
    abnormal = abs(difference) > threshold
    abnormal = abnormal[output_data.index]
    abnormal_indices = abnormal[abnormal].index
    return abnormal_indices

def abnormal_interval(output_data, abnormal_indices, hours):
    extended_indices = []
    for index in abnormal_indices:
        timestamp = index
        lower_bound = max(timestamp - timedelta(hours=hours), output_data.index[0])
        upper_bound = min(timestamp + timedelta(hours=hours + 1), output_data.index[-1] + timedelta(hours=1))
        extended_indices.extend([output_data.index[(output_data.index >= lower_bound) & (output_data.index < upper_bound)]])
    return extended_indices