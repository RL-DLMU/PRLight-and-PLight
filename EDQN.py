import torch
import torch.nn as nn
import torch.nn.functional as F
import os
import torch.optim as optim


class Embedding(nn.Module):
    def __init__(self, din, hidden_dim):
        super(Embedding, self).__init__()
        self.fc1 = nn.Linear(din, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, hidden_dim)

    def forward(self, x):
        x = F.relu(self.fc1(x))
        embedding = F.relu(self.fc2(x))
        return embedding


class AttModel(nn.Module):
    def __init__(self, din, hidden_dim, dout):
        super(AttModel, self).__init__()
        self.fcv = nn.Linear(din, hidden_dim)
        self.fck = nn.Linear(din, hidden_dim)
        self.fcq = nn.Linear(din, hidden_dim)
        self.fcout = nn.Linear(hidden_dim, dout)

    def forward(self, obs, adj):
        # obs [batch,n_agent,din]
        # adj [batch,n,neighbor,n_agent]

        _, num_node, _ = obs.size()
        hi = obs.unsqueeze(2)  # hi [batch,n_agent,1,din]
        obs_ = obs.unsqueeze(1)  # obs_ [batch,1,n_agent,din]
        obs_ = obs_.repeat(1, num_node, 1, 1)  # obs_ [batch,n,n_agent,din]
        hc = torch.matmul(adj, obs_)  # hc [batch,n,neighbor,din]

        query = F.relu(self.fcq(hi))  # query [batch,n_agent,1,hidden_dim]
        key = F.relu(self.fck(hc))  # key [batch,n_agent,neighbor,hidden_dim]
        key = key.permute(0, 1, 3, 2)  # key [batch,n_agent,hidden_dim,neighbor]
        att = F.softmax(torch.matmul(query, key), dim=-1)  # att [batch,n_agent,1,neighbor]
        value = F.relu(self.fcv(hc))  # value [batch,n_agent,neighbor,hidden_dim]
        out = F.relu(self.fcout(torch.matmul(att, value)))  # out [batch,n_agent,1,hidden_dim]
        out = out.squeeze(2)  # out [batch,n_agent,hidden_dim]

        return out


class Encoder(nn.Module):
    def __init__(self, num_inputs, hidden_dim):
        super(Encoder, self).__init__()

        self.mlp = Embedding(num_inputs, hidden_dim)
        self.att_1 = AttModel(hidden_dim, hidden_dim, hidden_dim)

    def forward(self, obs, adjs):

        h1 = self.mlp(obs)
        h2 = self.att_1(h1, adjs)

        return h2


class Decoder(nn.Module):
    def __init__(self, din, hidden_dim):
        super(Decoder, self).__init__()
        self.fc1 = nn.Linear(hidden_dim+1, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, din)

    def forward(self, h, actions):

        x = torch.cat((h, actions), dim=2)
        x1 = F.relu(self.fc1(x))
        x2 = F.relu(self.fc2(x1))

        return x2


class VAnet(torch.nn.Module):
    # V值，A值两个值头的网络架构——Dueling
    def __init__(self, din, dout):
        super(VAnet, self).__init__()
        self.f1 = nn.Linear(din, 128)
        self.fV = nn.Linear(128, 1)
        self.fA = nn.Linear(128, dout)

    # 将V值和A值合并成Q值返回
    def forward(self, x):

        x = F.relu(self.f1(x))
        A = self.fA(x)
        V = self.fV(x)
        avg_A = A.mean(2).unsqueeze(2)
        Q = V + A - avg_A   # Q值由V值和A值计算得到，减去均值防止方差过大，相当于标准化，保证训练稳定
        return Q


class EDcoder(torch.nn.Module):

    def __init__(self, encoder, decoder):
        super(EDcoder, self).__init__()
        self.encoder = encoder
        self.decoder = decoder

    def forward(self, obs, actions, adjs):
        h = self.encoder(obs, adjs)
        p = self.decoder(h, actions)
        return p


class EQnetwork(torch.nn.Module):

    def __init__(self, encoder, q_network):
        super(EQnetwork, self).__init__()
        self.encoder = encoder
        self.q_network = q_network

    def forward(self, obs, adjs):
        h = self.encoder(obs, adjs)
        q = self.q_network(h)
        return q


class EDQN(nn.Module):
    def __init__(self, num_inputs, hidden_dim, num_actions, device):
        super(EDQN, self).__init__()

        self.device = device
        self.encoder = Encoder(num_inputs, hidden_dim).to(device)
        self.decoder = Decoder(num_inputs, hidden_dim).to(device)
        self.q_net = VAnet(hidden_dim, num_actions).to(device)

    def forward(self, obs, actions, adjs):
        h = self.encoder(obs, adjs)
        p = self.decoder(h, actions)
        q = self.q_net(h)
        return q, p

    def load(self, file):
        encoder_file = os.path.join(file, 'encoder.pt')
        decoder_file = os.path.join(file, 'decoder.pt')
        q_file = os.path.join(file, 'q_net.pt')
        self.encoder.load_state_dict(torch.load(encoder_file))
        self.decoder.load_state_dict(torch.load(decoder_file))
        self.q_net.load_state_dict(torch.load(q_file))

    def save(self, path, cnt_round):
        if os.path.exists(path):
            torch.save(self.encoder.state_dict(), os.path.join(path, "%s.pt" % "encoder_round_{0}".format(cnt_round)))
            torch.save(self.decoder.state_dict(), os.path.join(path, "%s.pt" % "decoder_round_{0}".format(cnt_round)))
            torch.save(self.q_net.state_dict(), os.path.join(path, "%s.pt" % "q_net_round_{0}".format(cnt_round)))
        else:
            os.makedirs(path)