using System;


namespace DumbML {
    public class AddDims : Operation {
        int[] dims;
        int[] shapeActual;

        public AddDims(Operation input, int[] dims) {
            this.dims = (int[])dims.Clone();
            int newRank = input.shape.Length + dims.Length;


            HandleNegatives();
            CleanDims();
            ValidateDims();
            shapeActual = BuildShape(input.shape, shapeActual);
            BuildOp(shapeActual, input.dtype, input);


            void HandleNegatives() {
                for (int i = 0; i < this.dims.Length; i++) {
                    var v = this.dims[i];
                    if (this.dims[i] < 0) {
                        this.dims[i] = newRank + this.dims[i];
                        if (this.dims[i] < 0) {
                            throw new ArgumentException($"Dim is invalid: {v}");
                        }
                    }
                }
            }
            void CleanDims() {
                // remove repeats
                int numRepeat = 0;
                for (int i = 0; i < this.dims.Length; i++) {
                    // no repeat dims
                    for (int j = 0; j < i; j++) {
                        if (this.dims[i] == this.dims[j]) {
                            numRepeat++;
                        }
                    }
                }

                if (numRepeat == 0) {
                    return;
                }

                int[] newDims = new int[this.dims.Length - numRepeat];

                int ni = 0;
                for (int i = 0; i < newDims.Length; i++) {
                    bool valid = true;

                    for (int j = 0; j < ni; j++) {
                        if (newDims[j] == this.dims[i]) {
                            valid = false;
                        }
                    }

                    if (valid) {
                        newDims[ni] = this.dims[i];
                    }
                }

                this.dims = newDims;
            }
            void ValidateDims() {
                for (int i = 0; i < this.dims.Length; i++) {
                    if (this.dims[i] > newRank) {
                        throw new ArgumentException($"Dim is out of range: {this.dims[i]}");
                    }

                    // no repeat dims
                    for (int j = 0; j < i; j++) {
                        if (this.dims[i] == this.dims[j]) {
                            throw new ArgumentException($"Repeat dimensions not allowed");
                        }
                    }
                }
            }
        }


        public override void Forward(ITensorBuffer[] inputs, ITensorBuffer result) {
            shapeActual = BuildShape(inputs[0].shape, shapeActual);
            result.SetShape(shapeActual);

            BLAS.Engine.Compute.Copy(inputs[0], result, true);
        }

        public override Operation[] BuildBackwards(Operation[] inputs, Operation output, Operation error) {
            return new Operation[] {
                new Reshape(error, inputs[0])
            };
        }


        int[] BuildShape(int[] inputShape, int[] result) {
            result ??= new int[inputShape.Length + dims.Length];
            int ii = 0;
            for (int i = 0; i < result.Length; i++) {
                if (dims.Contains(i)) {
                    result[i] = 1;
                }
                else {
                    result[i] = inputShape[ii];
                    ii++;

                }
            }

            return result;
        }
    }

}
