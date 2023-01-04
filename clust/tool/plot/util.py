from io import BytesIO
import base64

def plt_to_image(plt):
    """
    Convert plt into real image
    Args:
        plt (matplotlib.pyplot):
    Return:
        image_base64 (image_base64)
    """
    # send images
    buf = BytesIO()
    plt.savefig(buf, format='png')
    image_base64 = base64.b64encode(buf.getvalue()).decode('utf-8').replace('\n', '')
    return image_base64