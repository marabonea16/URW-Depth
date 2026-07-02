# Copyright Niantic 2019. Patent Pending. All rights reserved.
#
# This software is licensed under the terms of the Monodepth2 licence
# which allows for non-commercial use only, the full terms of which are made
# available in the LICENSE file.



#depth_decoder
from __future__ import absolute_import, division, print_function

import numpy as np
import torch
import torch.nn as nn

from collections import OrderedDict
from layer import *
from .hr_layers import *
from scale_casa import scale_casa_HAM

class PWSA(nn.Module):
    def __init__(self, input_channel, output_channel):
        super(PWSA, self).__init__()

        self.Conv1x1 = Conv1x1(input_channel,input_channel)
        self.Res_block = ConvBlock(input_channel,input_channel)
        self.softmax = nn.Softmax(dim=1)
        #self.upsample = upsample()
        self.Conv1x1_out = Conv1x1(input_channel, output_channel)

    def forward(self, FD, FE):
        Sadd = (FD + FE)/2
        Satt = self.Res_block(FE)
        Satt = self.softmax(self.Conv1x1(Satt))
        Sscaled = Sadd * Satt
        S = self.Res_block(Sscaled)
        FD_out = upsample(self.Conv1x1_out(S))

        return FD_out


class FusionDecoder(nn.Module):
    def __init__(self, num_ch_enc, scales=range(4), num_output_channels=1, use_skips=True,
                 use_feature_suppression=False, gate_depth_input=True):
        super(FusionDecoder, self).__init__()

        self.num_output_channels = num_output_channels
        self.use_feature_suppression = use_feature_suppression
        # gate_depth_input=False: sigma e calculat (calibrat) dar NU mai gateaza
        # inputul lui dispconv - util cand vrei sigma interpretabil fara costul
        # de acuratete pe care suprima activa il introduce pe date curate/cunoscute.
        self.gate_depth_input = gate_depth_input
        self.scales = scales

        self.num_ch_enc = num_ch_enc        #features in encoder, [16,64,128,160,320]

        # decoder
        self.convs = OrderedDict()

        self.convs[("parallel_conv"), 0, 0] = ConvBlock(self.num_ch_enc[0], self.num_ch_enc[0])
        self.convs[("parallel_conv"), 0, 1] = ConvBlock(self.num_ch_enc[1], self.num_ch_enc[1])
        self.convs[("parallel_conv"), 0, 2] = ConvBlock(self.num_ch_enc[2], self.num_ch_enc[2])
        self.convs[("parallel_conv"), 0, 3] = ConvBlock(self.num_ch_enc[3], self.num_ch_enc[3])

        self.convs[("conv1x1", 0, 2_1)] = ConvBlock1x1(self.num_ch_enc[1]+self.num_ch_enc[0], self.num_ch_enc[0])
        self.convs[("conv1x1", 0, 3_2)] = ConvBlock1x1(self.num_ch_enc[2]+self.num_ch_enc[1], self.num_ch_enc[1])

        self.convs[("conv1x1", 0, 4_3)] = ConvBlock1x1(self.num_ch_enc[3]+self.num_ch_enc[2], self.num_ch_enc[2])

        self.convs[("attention", 4)] = fSEModule(self.num_ch_enc[3], self.num_ch_enc[4])



        self.convs[("parallel_conv"), 1, 0] = ConvBlock(self.num_ch_enc[0], self.num_ch_enc[0])
        self.convs[("parallel_conv"), 1, 1] = ConvBlock(self.num_ch_enc[1], self.num_ch_enc[1])
        self.convs[("parallel_conv"), 1, 2] = ConvBlock(self.num_ch_enc[2], self.num_ch_enc[2])

        self.convs[("conv1x1", 1, 2_1)] = ConvBlock1x1(self.num_ch_enc[1]+self.num_ch_enc[0], self.num_ch_enc[0])
        self.convs[("conv1x1", 1, 3_2)] = ConvBlock1x1(self.num_ch_enc[2]+self.num_ch_enc[1], self.num_ch_enc[1])
        self.convs[("attention", 3)] = fSEModule(self.num_ch_enc[2], self.num_ch_enc[3])



        self.convs[("parallel_conv"), 2, 0] = ConvBlock(self.num_ch_enc[0], self.num_ch_enc[0])
        self.convs[("parallel_conv"), 2, 1] = ConvBlock(self.num_ch_enc[1], self.num_ch_enc[1])
        self.convs[("conv1x1", 2, 2_1)] = ConvBlock1x1(self.num_ch_enc[1]+self.num_ch_enc[0], self.num_ch_enc[0])
        self.convs[("attention", 2)] = fSEModule(self.num_ch_enc[1], self.num_ch_enc[2])

        #待定
        self.convs[("parallel_conv"), 3, 0] = ConvBlock(self.num_ch_enc[0], self.num_ch_enc[0])


        self.convs[("attention", 1)] = fSEModule(self.num_ch_enc[0], self.num_ch_enc[1])


        self.convs[("dispconv", 0)] = Conv3x3(64, self.num_output_channels)
        self.convs[("uncertconv", 0)] = Conv3x3(64, self.num_output_channels)
        # bias=0 -> sigmoid(0)=0.5: sigma porneste la mijloc, evita sigma collapse
        nn.init.constant_(self.convs[("uncertconv", 0)].conv.bias, 0.0)

        # cap global de detectie a corupiei: distinge "imagine global curata cu
        # regiuni local dificile" de "imagine global corupta cu vreme", folosind
        # eticheta reala din pipeline-ul de augmentare (supravegheat, nu auto-
        # supervizat ca sigma). g gateaza suprima de caracteristici - pe imagini
        # curate g~0 => d_refined~d (adancime neafectata), pe imagini corupte
        # g~1 => suprima functioneaza normal, ghidata de sigma local.
        # IMPORTANT: citeste din imaginea RGB bruta (3 canale), nu din feature-ul
        # tarziu `d` - encoder+decoder sunt antrenate sa fie INVARIANTE la vreme
        # (asta e scopul robustetii la vreme!), deci semnalul de "e corupta" e
        # deja eliminat din `d` (verificat empiric: semnal/zgomot 0.09 pe `d` vs
        # 3.6-3.9 pe imaginea bruta).
        # Input = medie + std per canal (6 valori), nu doar medie (3): media
        # singura prinde ceata bine dar e aproape oarba la zapada/ploaie usoara;
        # std-ul prinde desaturarea/scaderea de contrast pe care media o rateaza.
        self.corruption_head = nn.Sequential(
            nn.Linear(6, 16),
            nn.ReLU(inplace=True),
            nn.Linear(16, 1),
        )
        nn.init.constant_(self.corruption_head[-1].bias, 0.0)




        self.decoder = nn.ModuleList(list(self.convs.values()))
        self.sigmoid = nn.Sigmoid()


        self.CASA = scale_casa_HAM([64, 128, 160, 320])


    def FusionConv(self, conv, high_feature, low_feature, scale_fac):

        high_features = [updown_sample(high_feature, scale_fac)]  #test

        high_features.append(low_feature)

        high_features = torch.cat(high_features, 1)


        return conv(high_features)



    def forward(self, input_feature, raw_image=None):
        self.outputs = {}
        self._raw_image = raw_image  # folosit de corruption_head, vezi mai jos




        input_features = self.CASA(input_feature)

        e4 = input_features[4]
        e3 = input_features[3]
        e2 = input_features[2]
        e1 = input_features[1]
        e0 = input_features[0]


        d0_1 = self.convs[("parallel_conv"), 0, 0](e0)
        d0_2 = self.convs[("parallel_conv"), 0, 1](e1)
        d0_3 = self.convs[("parallel_conv"), 0, 2](e2)
        d0_4 = self.convs[("parallel_conv"), 0, 3](e3)


        d05_4 = self.convs[("attention", 4)](e4, d0_4)
        d04_3 = self.FusionConv(self.convs[("conv1x1", 0, 4_3)], d0_4, d0_3, 2)
        d03_2 = self.FusionConv(self.convs[("conv1x1", 0, 3_2)], d0_3, d0_2, 2)

        d02_1 = self.FusionConv(self.convs[("conv1x1", 0, 2_1)], d0_2, d0_1, 2)


        d1_1 = self.convs[("parallel_conv"), 1, 0](d02_1)
        d1_2 = self.convs[("parallel_conv"), 1, 1](d03_2)

        d1_3 = self.convs[("parallel_conv"), 1, 2](d04_3)

        d14_3 = self.convs[("attention", 3)](d05_4, d1_3)
        d13_2 = self.FusionConv(self.convs[("conv1x1", 1, 3_2)], d1_3, d1_2, 2)

        d12_1 = self.FusionConv(self.convs[("conv1x1", 1, 2_1)], d1_2, d1_1, 2)



        d2_1 = self.convs[("parallel_conv"), 2, 0](d12_1)
        d2_2 = self.convs[("parallel_conv"), 2, 1](d13_2)

        d23_2 = self.convs[("attention", 2)](d14_3, d2_2)
        d22_1 = self.FusionConv(self.convs[("conv1x1", 2, 2_1)], d2_2, d2_1, 2)



        d3_0 = self.convs[("parallel_conv"), 3, 0](d22_1)
        d32_1 = self.convs[("attention", 1)](d23_2, d3_0)


        d = self.convs[("parallel_conv"), 3, 0](d32_1)


        d = updown_sample(d, 2)
        self._last_d = d  # expus pentru experimentare externa (vezi test_nogate_checkpoint.py)


        # d.detach(): izoleaza capul de incertitudine de backbone-ul partajat cu
        # dispconv. Fara detach, pierderile de incertitudine (calibrare MSE,
        # smoothness, consistenta cross-modal) propaga gradient inapoi prin d
        # si reshape-uiesc feature-urile folosite si pentru adancime, degradand
        # acuratetea depth chiar daca masca/calibrarea functioneaza "corect".
        self.outputs[("uncert", 0)] = self.convs[("uncertconv", 0)](d.detach())  # log-variance (raw, no activation)
        # cap global de detectie a corupiei - citeste din imaginea RGB bruta
        # (vezi nota din __init__: feature-ul tarziu `d` e invariant la vreme,
        # nu mai poarta semnalul necesar). Daca raw_image nu e disponibil
        # (apeluri vechi/alte scripturi), g ramane neutru (0.5) si gating-ul
        # global e dezactivat - comportament identic cu varianta fara cap nou.
        if self._raw_image is not None:
            stats = torch.cat([
                self._raw_image.mean(dim=(2, 3)),
                self._raw_image.std(dim=(2, 3)),
            ], dim=1)  # [B,6]
            self.outputs[("corrupt_logit", 0)] = self.corruption_head(stats)  # [B,1]
        else:
            self.outputs[("corrupt_logit", 0)] = torch.zeros(d.shape[0], 1, device=d.device)
        if self.use_feature_suppression and getattr(self, "gate_depth_input", True):
            # uncertainty-guided feature suppression, gateata global:
            # sigma (local) suprima feature-urile, dar doar pe imagini pe care
            # capul de corupie le considera probabil corupte (g~1); pe imagini
            # curate g~0 => d_refined~d, indiferent ce arata sigma local.
            sigma = self.sigmoid(self.outputs[("uncert", 0)])
            gate_temp = getattr(self, "gate_temperature", 1.0)
            g = self.sigmoid(self.outputs[("corrupt_logit", 0)] * gate_temp).view(-1, 1, 1, 1)
            if getattr(self, "hard_gate", False):
                # gate dur (0/1) la inferenta: elimina costul rezidual de
                # suprimare pe imagini clar curate (g~0.13, nu exact 0) -
                # testat empiric ca alternativa fara reantrenare.
                g = (g > 0.5).float()
            d_refined = d * (1.0 - g.detach() * sigma.detach())
            self.outputs[("disp", 0)] = self.sigmoid(self.convs[("dispconv", 0)](d_refined))
        elif self.use_feature_suppression:
            # suprimare pura sigma fara gate corruption_head (d*(1-sigma))
            # folosita la evaluarea Fix4/URW-Depth-Weather, antrenat inainte de
            # adaugarea corruption_head.
            sigma = self.sigmoid(self.outputs[("uncert", 0)])
            d_refined = d * (1.0 - sigma.detach())
            self.outputs[("disp", 0)] = self.sigmoid(self.convs[("dispconv", 0)](d_refined))
        else:
            self.outputs[("disp", 0)] = self.sigmoid(self.convs[("dispconv", 0)](d))


        return self.outputs






