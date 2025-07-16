# Differences Between FCN Variants and Performance Analysis

## Architectural Differences

### Skip Connections and Feature Fusion

#### **FCN-32s:**

- No skip connections
- Uses only high-level features from the final layer of VGG16
- Focuses solely on semantic information but lacks spatial precision

#### **FCN-16s:**

- Incorporates one skip connection from pool4 (mid-level features)
- Combines coarse semantic information with mid-level spatial details
- Enhances boundary delineation compared to FCN-32s

#### **FCN-8s:**

- Incorporates two skip connections from both pool4 and pool3 layers
- Integrates high-level semantic features with mid and low-level spatial details
- Provides the finest spatial resolution and boundary precision

### Upsampling Strategy

#### **FCN-32s:**

- Single transposed convolution upsampling by 32x
- Simplest approach but results in coarser output

#### **FCN-16s:**

- Two-stage upsampling process
- First upsamples by 2x and fuses with pool4 features
- Then upsamples by 16x to original resolution (total 32x)

#### **FCN-8s:**

- Progressive three-stage upsampling
- First upsamples by 2x and fuses with pool4
- Further upsamples by 2x and fuses with pool3
- Final 8x upsampling to reach original resolution (total 32x)

## Performance Results

### Frozen Backbone  [FCN_1](FCN_1.ipynb)

| Model       | Validation mIoU |
|-------------|-----------------|
| FCN-32s     | 0.7820          |
| FCN-16s     | 0.7883          |
| FCN-8s      | 0.8151          |

### Fine-tuned Backbone  [FCN_2](FCN_2.ipynb)

| Model       | Validation mIoU |
|-------------|-----------------|
| FCN-32s-FT  | 0.7809          |
| FCN-16s-FT  | 0.8054          |
| FCN-8s-FT   | 0.8275          |

## Key Findings

### Skip Connection Impact:

- Performance improves with additional skip connections
    - Frozen backbone: FCN-32s → FCN-8s (+3.31%)
    - Fine-tuned backbone: FCN-32s-FT → FCN-8s-FT (+4.66%)

### Fine-tuning Effects:

- **FCN-32s:** Fine-tuning slightly decreased performance (-0.11%)
- **FCN-16s:** Fine-tuning improved performance (+1.71%)
- **FCN-8s:** Fine-tuning improved performance (+1.24%)

### Best Performance:

- **FCN-8s-FT** achieved highest mIoU (0.8275)
    - Combines benefits of multiple skip connections with task-specific feature adaptation

## Conclusion

The **FCN-8s** architecture consistently outperforms other variants regardless of backbone training strategy, highlighting the importance of incorporating multi-scale features. Fine-tuning provides significant benefits for architectures with skip connections but showed no advantage for FCN-32s. The optimal configuration is **FCN-8s with a fine-tuned backbone**, which effectively balances pretrained feature utilization with task-specific adaptation.
