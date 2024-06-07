import plotly.express as px
import plotly.graph_objects as go


def plot_loss(train_loss: list[float], val_loss: list[float]) -> go.Figure:
    epochs = list(range(1, len(train_loss)+1))
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=epochs, y=train_loss, mode='lines', name='Train loss'))
    fig.add_trace(go.Scatter(x=epochs, y=val_loss, mode='lines', name='Validation loss'))

    fig.update_layout(
        title='Loss',
        xaxis_title='Epochs',
        yaxis_title='Value',
        template='plotly_white'
    )
    return fig


def plot_metrics(
    train_statistics: dict[str, list[float]],
    val_statistics: dict[str, list[float]]
) -> go.Figure:
    fig = go.Figure()
    for key, value in train_statistics.items():
        epochs = list(range(1, len(value)+1))
        fig.add_trace(go.Scatter(x=epochs, y=value, mode='lines', name=f'Train {key}'))
    for key, value in val_statistics.items():
        epochs = list(range(1, len(value) + 1))
        fig.add_trace(go.Scatter(x=epochs, y=value, mode='lines', name=f'Validation {key}'))
    fig.update_layout(
        title='Metrics',
        xaxis_title='Epochs',
        yaxis_title='Value',
        template='plotly_white'
    )
    return fig
