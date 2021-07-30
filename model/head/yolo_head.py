import torch.nn as nn
import torch


class Yolo_head(nn.Module):
    def __init__(self, nC, anchors, stride):
        super(Yolo_head, self).__init__()

        self.__anchors = anchors
        self.__nA = len(anchors)
        self.__nC = nC
        self.__stride = stride
        print("Yolo_head __init__:",self.__anchors , self.__nA, self.__nC, self.__stride  );

    def forward(self, p):
        bs, nG = p.shape[0], p.shape[-1]

        print("Yolo_head forward bs ",bs, " nG: ", nG );
        p = p.view(bs, self.__nA, 5 + self.__nC, nG, nG)
        print("Yolo_head forward p view ", p.shape );
        p = p.permute(0, 3, 4, 1, 2)
        print("Yolo_head forward p permute", p.shape );
        p_de = self.__decode(p.clone())
        print("Yolo_head forward p_de ", p_de.shape );
        return (p, p_de)


    def __decode(self, p):
        batch_size, output_size = p.shape[:2]

        print("Yolo_head forward __decode1 batch_size ", batch_size, " output_size:",output_size );
        device = p.device
        stride = self.__stride
        anchors = (1.0 * self.__anchors).to(device)

        print("Yolo_head forward __decode1 device ", device, " stride:",stride, " anchors:", anchors.shape , anchors);

        conv_raw_dxdy = p[:, :, :, :, 0:2]
        conv_raw_dwdh = p[:, :, :, :, 2:4]
        conv_raw_conf = p[:, :, :, :, 4:5]
        conv_raw_prob = p[:, :, :, :, 5:]


        y = torch.arange(0, output_size).unsqueeze(1).repeat(1, output_size)

        print("Yolo_head forward __decode2 y::: ",torch.arange(0, output_size), torch.arange(0, output_size).unsqueeze(1),  y);
        x = torch.arange(0, output_size).unsqueeze(0).repeat(output_size, 1)
        print("Yolo_head forward __decode2 x::: ",torch.arange(0, output_size), torch.arange(0, output_size).unsqueeze(0),  x);
        grid_xy = torch.stack([x, y], dim=-1)
        grid_xy = grid_xy.unsqueeze(0).unsqueeze(3).repeat(batch_size, 1, 1, 3, 1).float().to(device)

        pred_xy = (torch.sigmoid(conv_raw_dxdy) + grid_xy) * stride
        print("Yolo_head forward __decode2 pred_xy pre ", (torch.sigmoid(conv_raw_dxdy) + grid_xy), " pred_xy:", pred_xy);
        pred_wh = (torch.exp(conv_raw_dwdh) * anchors) * stride

        pred_xywh = torch.cat([pred_xy, pred_wh], dim=-1)
        pred_conf = torch.sigmoid(conv_raw_conf)
        pred_prob = torch.sigmoid(conv_raw_prob)
        pred_bbox = torch.cat([pred_xywh, pred_conf, pred_prob], dim=-1)

        print("Yolo_head forward __decode2 not self.training ", not self.training); 

        if not self.training:
            aa = pred_bbox.view(-1, 5 + self.__nC)
            print("Yolo_head forward __decode2 not self.training pred_bbox", aa.shape); 
        else: 
            print("Yolo_head forward __decode2 self.training pred_bbox", pred_bbox.shape); 

        return pred_bbox.view(-1, 5 + self.__nC) if not self.training else pred_bbox
