# -*- encoding: utf-8 -*-
# @Author: Yihuai Lan
# @Time: 2021/08/29 21:49:07
# @File: stack_machine.py


import sympy
import torch

class OPERATIONS:
    def __init__(self, out_symbol2idx):
        self.NOOP = -1
        self.GEN_VAR = -2
        self.PAD = out_symbol2idx['<PAD>']
        self.ADD = out_symbol2idx['+']
        self.SUB = out_symbol2idx['-']
        self.MUL = out_symbol2idx['*']
        self.DIV = out_symbol2idx['/']
        self.POWER = out_symbol2idx['^']
        self.RAW_EQL = out_symbol2idx['='] if '=' in out_symbol2idx else -1
        self.BRG = out_symbol2idx['<BRG>'] if '<BRG>' in out_symbol2idx else -1
        self.EQL = out_symbol2idx['<EOS>']
        self.N_OPS = out_symbol2idx['NUM_0']
        #self.N_OPS = self.BRG+1 if self.POWER < self.BRG else self.POWER+1
        #self.N_OPS = self.POWER+1

class StackMachine:
    def __init__(self, operations, constants, embeddings, bottom_embedding, dry_run=False):
        """
        Args:
            constants (list): Value of numbers.
            embeddings (tensor): Tensor of shape [len(constants), dim_embedding].
                Embedding of the constants.
            bottom_embedding (teonsor): Tensor of shape (dim_embedding,). The
                embeding to return when stack is empty.
        """
        self._operands = list(constants)
        self._embeddings = [embedding for embedding in embeddings]
        self.operations = operations

        # number of unknown variables
        self._n_nuknown = 0

        # stack which stores (val, embed) tuples
        self._stack = []

        # equations got from applying `=` on the stack
        self._equations = []
        self.stack_log = []
        self.stack_log_index = []

        # functions operate on value
        self._val_funcs = {
            self.operations.ADD: sympy.Add,
            self.operations.SUB: lambda a, b: sympy.Add(-a, b),
            self.operations.MUL: sympy.Mul,
            self.operations.DIV: lambda a, b: sympy.Mul(1/a, b),
            self.operations.POWER: lambda a, b: sympy.POW(a, b)
        }
        self._op_chars = {
            self.operations.ADD: '+',
            self.operations.SUB: '-',
            self.operations.MUL: '*',
            self.operations.DIV: '/',
            self.operations.POWER: '^',
            self.operations.RAW_EQL: '=',
            self.operations.BRG: '<BRG>',
            self.operations.EQL: '<EOS>'
        }
        #print(self._operands, self._op_chars); exit()

        self._bottom_embed = bottom_embedding

        if dry_run:
            self.apply = self.apply_embed_only

    def add_variable(self, embedding):
        """ Tell the stack machine to increase the number of nuknown variables
            by 1.

        Args:
            embedding (torch.Tensor): Tensor of shape (dim_embedding). Embedding
                of the unknown varialbe.
        """
        var = sympy.Symbol('x{}'.format(self._n_nuknown))
        self._operands.append(var)
        self._embeddings.append(embedding)
        self._n_nuknown += 1

        # self.stack_log.append(var)
        # self.stack_log_index.append(OPERATIONS.GEN_VAR)

    def push(self, operand_index):
        """ Push var to stack.

        Args:
            operand_index (int): Index of the operand. If index >= number of constants, then it implies a variable is pushed.
        
        Returns:
            torch.Tensor: Simply return the pushed embedding.
        """
        self._stack.append((self._operands[operand_index],
                            self._embeddings[operand_index]))
        self.stack_log.append(self._operands[operand_index])
        #print('self.stack_log', self.stack_log, operand_index, self._operands, self._op_chars, self.operations.N_OPS); exit()
        self.stack_log_index.append(operand_index + self.operations.N_OPS) #
        return self._embeddings[operand_index]

    def apply_embed_only(self, operation, embed_res):
        """ Apply operator on stack with embedding operation only.

        Args:
            operator (mwptoolkit.module.Environment.stack_machine.OPERATION): One of
                - OPERATIONS.ADD
                - OPERATIONS.SUB
                - OPERATIONS.MUL
                - OPERATIONS.DIV
                - OPERATIONS.EQL
            embed_res (torch.FloatTensor): Resulted embedding after transformation, with size (dim_embedding,).
        
        Returns:
            torch.Tensor: embedding on the top of the stack.
        """
        if len(self._stack) < 2: return self._bottom_embed
        val1, embed1 = self._stack.pop()
        val2, embed2 = self._stack.pop()
        if operation not in [self.operations.RAW_EQL, self.operations.BRG]:
            # calcuate values in the equation
            val_res = None
            # transform embedding
            self._stack.append((val_res, embed_res))

        self.stack_log.append(self._op_chars[operation])
        self.stack_log_index.append(operation)

        if len(self._stack) > 0:
            return self._stack[-1][1]
        else:
            return self._bottom_embed

    def apply_eql(self, operation):

        self.stack_log.append(self._op_chars[operation])
        self.stack_log_index.append(operation)
        return self._bottom_embed

    def get_solution(self):
        """ Get solution. If the problem has not been solved, return None.

        Returns:
            list: If the problem has been solved, return result from sympy.solve. If not, return None.
        """

        if self._n_nuknown == 0:
            #print('return None', self._equations)
            return None

        try:
            #print('self._equations', self._equations)
            root = sympy.solve(self._equations)
            #print(self._equations, root)
            for i in range(self._n_nuknown):
                if self._operands[-i - 1] not in root:
                    #print('return None')
                    return None

            return root
        except:
            return None

    def get_top2(self):
        """ Get the top 2 embeddings of the stack.

        Return:
            torch.Tensor: Return tensor of shape (2, embed_dim).
        """
        if len(self._stack) >= 2:
            return torch.stack([self._stack[-1][1],
                                self._stack[-2][1]], dim=0)
        elif len(self._stack) == 1:
            #print(self._stack[-1][1], self._bottom_embed)
            return torch.stack([self._stack[-1][1],
                                self._bottom_embed], dim=0)
        else:
            return torch.stack([self._bottom_embed,
                                self._bottom_embed], dim=0)

    def get_height(self):
        """ Get the height of the stack.

        Return:
            int: height.
        """
        return len(self._stack)

    def get_stack(self):
        return [self._bottom_embed] + [s[1] for s in self._stack]
